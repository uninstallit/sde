import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import time
import os

tfb = tfp.bijectors
tfd = tfp.distributions

# tf.config.run_functions_eagerly(True)

tf.config.optimizer.set_jit(True)
os.environ[
    "TF_XLA_FLAGS"
] = "--tf_xla_cpu_global_jit /mnt/c/Users/Edvin/Desktop/sde-image/sde-image-generation/diffusion-vae.py"


class ConditionedDiffusionBijector(tfb.Bijector):
    def __init__(
        self,
        mu,
        sigma,
        ti,
        dt,
        paths,
        density,
        validate_args=False,
        name="cond_diffusion",
    ):
        super(ConditionedDiffusionBijector, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name,
            is_constant_jacobian=True,
        )

        self.event_ndim = 1
        self.mu = mu
        self.sigma = sigma
        self.paths = paths
        self.time = ti
        self.ti = tf.repeat(ti, self.paths, axis=0)
        self.dt = dt
        self.density = density

        self.prev_x = None
        self.next_y = None
        self.point = None

    def _forward(self, x):
        self.prev_x = x
        next_x = (
            x
            + (self.mu(x, self.ti) * self.dt)
            + (
                self.sigma(x, self.ti)
                * tf.math.sqrt(self.dt)
                * np.random.randn(self.paths)
            )
        )
        return next_x

    def _inverse(self, y):
        next_y = (
            y
            + (
                (
                    self.mu(y, self.ti)
                    - (self.sigma(y, self.ti) * self._forward_log_det_jacobian(y))
                )
                * self.dt
            )
            + (
                self.sigma(y, self.ti)
                * tf.math.sqrt(self.dt)
                * np.random.randn(self.paths)
            )
        )
        self.next_y = next_y
        return next_y

    # def _inverse_log_det_jacobian(self, y, t):
    #     return -self._forward_log_det_jacobian(self._inverse(y, t))

    def _forward_log_det_jacobian(self, x):
        self.point = tf.transpose(tf.concat([[x], [self.ti]], axis=0))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.point)
            dlogf_dxt = tape.gradient(
                tf.math.log(self.density(self.point, training=True)), self.point
            )
        return dlogf_dxt[:, 0]


# model
density_inputs = keras.Input(shape=(2,))
xl = keras.layers.Dense(64, activation="relu")(density_inputs)
xl = keras.layers.Dense(32, activation="sigmoid")(xl)
xl = keras.layers.Dense(64, activation="sigmoid")(xl)
density_outputs = keras.layers.Dense(1, activation="linear")(xl)
model = keras.Model(density_inputs, density_outputs, name="density")
model.compile(optimizer="Adam", loss="mse")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_tracker = keras.metrics.Mean(name="loss")

mix = 0.3
bimix_gauss = tfd.Mixture(
    cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
    components=[
        tfd.Normal(loc=-1.0, scale=0.5),
        tfd.Normal(loc=+1.0, scale=0.5),
    ],
)

normal = tfp.distributions.Normal(
    0.0, 5.0, validate_args=False, allow_nan_stats=True, name="Normal"
)


@tf.function
def reconstruction(prev_x, next_y):
    mae_loss = tf.reduce_mean(tf.math.abs(prev_x - next_y))
    return mae_loss


@tf.function
def kolmogorov(point, mu, sigma):
    x = point[:, 0]
    t = point[:, 1]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(point)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(point)
            f = model(point, training=True)
            df_dxt = tape1.gradient(f, point)
            d2f_dxt2 = tape1.gradient(df_dxt, point)

            # forward kolmogorov
            # df_dt = - mu * df_dx + 0.5 * sig^2 * d2f_dx2
            klm_loss = tf.reduce_sum(
                tf.math.abs(
                    df_dxt[:, 1]
                    + mu(x, t) * df_dxt[:, 0]
                    - 0.5 * tf.math.square(sigma(x, t)) * d2f_dxt2[:, 0]
                )
            )
    return klm_loss


# @tf.function
def train_step(bijector, paths, mu, sigma):
    x0 = bimix_gauss.sample(paths)
    xt = bijector._forward(x0)

    with tf.GradientTape(persistent=True) as tape:
        x0_bar = bijector._inverse(xt.numpy())  # <- breaks if you pass tensors

        prev_x = bijector.bijectors[0].prev_x
        next_y = bijector.bijectors[0].next_y
        time = bijector.bijectors[0].time
        point = bijector.bijectors[0].point

        mae_loss = reconstruction(prev_x, next_y)
        klm_loss = kolmogorov(point, mu, sigma)
        loss = mae_loss + klm_loss

        if (len(bijector.bijectors) % 100 == 0):
            print("time: {:.2f} - loss: {:.2f}".format(time, loss.numpy()))

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def main():

    mu = lambda x, t: 0.0
    sigma = lambda x, t: 1.5
    paths = 1000
    tmin = 0.0
    tmax = 5.0
    dt = 0.01
    steps = int((tmax - tmin) / dt)
    t = tf.linspace(tmin, tmax, int(steps + 1))

    epochs = 100
    for epoch in range(epochs):
        start_time = time.time()

        bijectors = []

        for ti in t[1:]:
            bijectors.append(
                ConditionedDiffusionBijector(
                    mu,
                    sigma,
                    ti,
                    dt,
                    paths,
                    model,
                )
            )

            bijector = tfb.Chain(list(reversed(bijectors)))
            loss_value = train_step(bijector, paths, mu, sigma)

        loss_tracker.update_state(loss_value)
        train_acc = loss_tracker.result()
        elapsed = time.time() - start_time
        loss_tracker.reset_states()

        print(
            "\nEpoch: {} - Elapsed: {:.2f} seconds - Loss: {:.2f}\n".format(epoch, elapsed, train_acc)
        )

    x0 = bimix_gauss.sample(paths)
    xt = bijector._forward(x0)
    x0_bar = bijector._inverse(xt.numpy())
    plt.hist(x0.numpy(), 100, alpha=0.5, label="x0")
    plt.hist(xt.numpy(), 100, alpha=0.5, label="xt")
    plt.hist(x0_bar.numpy(), 100, alpha=0.5, label="x0_bar")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
