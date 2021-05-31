import tensorflow as tf
from tensorflow import keras
import numpy as np


class ForwardDiffusion(tf.keras.layers.Layer):
    def __init__(self, mu, sigma, tmin, tmax, dt, length):
        super(ForwardDiffusion, self).__init__()
        self.mu = mu
        self.sigma = sigma

        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt
        self.time = tf.constant(np.arange(tmin, tmax, dt))
        self.steps = len(self.time)

        self.length = length
        self.ones = tf.ones((length,))

    def call(self, x):
        diffusion = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        diffusion = diffusion.write(0, x)

        for step in tf.data.Dataset.range(1, self.steps, 1):
            t = self.time[step]
            x = (
                x
                + (self.mu(x, t) * self.ones) * self.dt
                + (self.sigma(x, t) * self.ones)
                * tf.math.sqrt(self.dt)
                * tf.random.normal((self.length,))
            )
            diffusion = diffusion.write(tf.cast(step, dtype=tf.int32), x)
        return tf.transpose(diffusion.stack(), perm=[1, 0, 2])


class ConditionDiffusion(keras.Model):
    def __init__(
        self,
        diffusion,
        score_func,
        mu,
        sigma,
        tmin,
        tmax,
        dt,
        length,
        batch_size,
        **kwargs
    ):
        super(ConditionDiffusion, self).__init__(**kwargs)
        self.diffusion = diffusion
        self.score_func = score_func
        self.mu = mu
        self.sigma = sigma

        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt
        self.time = tf.constant(np.arange(tmin, tmax, dt), dtype=tf.float32)
        self.steps = len(self.time)

        self.length = length
        self.ones = tf.ones((length,))
        self.batch_size = batch_size

        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, forward_diffusion):
        reverse_diffusion = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        x = tf.squeeze(forward_diffusion[:, -1:])
        reverse_diffusion = reverse_diffusion.write(0, x)

        for step in tf.data.Dataset.range(self.steps - 1, 0, -1):
            t = self.time[step]
            batch_t = tf.repeat(t[tf.newaxis, tf.newaxis], self.batch_size, axis=0)
            x = (
                x
                + (
                    self.mu(x, t) * self.ones
                    - tf.math.square(self.sigma(x, t))
                    * self.ones
                    * self.score_func(
                        tf.concat([x, batch_t], axis=1),
                        training=True,
                    )
                )
                * self.dt
                + (self.sigma(x, t) * self.ones)
                * tf.math.sqrt(self.dt)
                * tf.random.normal((self.length,))
            )
            reverse_diffusion = reverse_diffusion.write(
                tf.cast(step, dtype=tf.int32), x
            )
        reverse_diffusion = tf.transpose(reverse_diffusion.stack(), perm=[1, 0, 2])
        return reverse_diffusion

    def train_step(self, data):
        images = data

        with tf.GradientTape(persistent=True) as tape:

            forward_diffusion = self.diffusion(images)
            reverse_diffusion = self(forward_diffusion, training=True)
            reconstruction_loss = tf.reduce_sum(
                keras.losses.MSE(forward_diffusion, reverse_diffusion), axis=1
            )

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(reconstruction_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }


def main():

    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    img_num = images.shape[0]
    img_dim = images.shape[1]
    length = img_dim * img_dim
    images = np.reshape(images, (img_num, length)).astype(np.float32)

    mu = lambda x, t: 0.5
    sigma = lambda x, t: 10.0

    tmin = 0
    tmax = 1
    dt = 0.01

    batch_size = 1
    epochs = 1

    diffusion = ForwardDiffusion(mu, sigma, tmin, tmax, dt, length)

    score_inputs = keras.layers.Input(shape=(length + 1,))
    x = keras.layers.Dense(100, activation="tanh")(score_inputs)
    x = keras.layers.Dense(100, activation="tanh")(x)
    x = keras.layers.Dense(100, activation="tanh")(x)
    score_outputs = keras.layers.Dense(length, activation="linear")(x)
    score_function = keras.Model(score_inputs, score_outputs, name="predictor")
    score_function.build(score_inputs)

    conditioned_diffusion = ConditionDiffusion(
        diffusion, score_function, mu, sigma, tmin, tmax, dt, length, batch_size
    )

    # !does not build - too big
    conditioned_diffusion.build(input_shape=(None, length))

    conditioned_diffusion.compile(optimizer=keras.optimizers.Adam())
    conditioned_diffusion.fit(
        images,
        epochs=epochs,
        batch_size=batch_size,
        workers=-1,
    )


if __name__ == "__main__":
    main()
