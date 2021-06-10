import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import training

tfd = tfp.distributions


class DiffusionSampler(tf.keras.utils.Sequence):
    def __init__(self, paths, epochs, *args, **kwargs):
        self.paths = paths
        self.epochs = epochs

        self.initial_distribution = tfd.Normal(loc=0, scale=1)

        pi = np.array([0.3, 0.7], dtype=np.float32)
        mu = np.array([-2, 5], dtype=np.float32)
        sigma = np.array([2, 2], dtype=np.float32)
        self.final_distribution = tfd.Mixture(
            cat=tfd.Categorical(probs=pi),
            components=[tfd.Normal(loc=m, scale=s) for m, s in zip(mu, sigma)],
        )

    def __len__(self):
        return self.epochs

    def __getitem__(self, index):
        s = tfd.Sample(self.initial_distribution, sample_shape=self.paths)
        x0 = s.sample()

        s = tfd.Sample(self.final_distribution, sample_shape=self.paths)
        xt = s.sample()
        return x0, xt

    # def on_epoch_end(self):
    # option method to run some logic at the end of each epoch: e.g. reshuffling


class Diffusion(tf.keras.Model):
    def __init__(self, encoder, paths, tmin, tmax, dt, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.encoder = encoder
        self.paths = paths
        self.tmin = tmin
        self.tmin = tmax
        self.dt = dt

        self.ones = tf.ones([self.paths], dtype=tf.float32)

    def call(self, x0):
        # x = self.diffusion(x0)
        return x0

    def train_step(self, data):
        x0, xt = data

        steps = tf.data.Dataset.from_tensor_slices(
            np.arange(self.tmin, self.tmin, self.dt)
        )

        with tf.GradientTape() as tape:

            x = x0
            for step in steps:
                t = tf.squeeze(tf.cast(step, tf.float32) * self.dt * self.ones)
                points = tf.transpose(tf.stack([x, t], 0))

                mu, sigma = self.encoder(points, training=True)
                mu = tf.squeeze(mu)
                sigma = tf.squeeze(sigma)

                dx = mu * self.ones * self.dt + sigma * self.ones * tf.math.sqrt(
                    self.dt
                ) * tf.random.normal([self.paths], dtype=tf.float32)
                x = x + dx

            reconstruction_loss = tf.reduce_sum(keras.losses.MSE(xt, x), axis=-1)

        grads = tape.gradient(reconstruction_loss, self.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.self.encoder.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {"reconstruction_loss": self.reconstruction_loss_tracker.result()}


def main():

    # x = np.linspace(-5, 5, 100)
    # plt.plot(x, normal_dist.prob(x).numpy())
    # plt.show()

    latent_dim = 1
    paths = 10
    epochs = 100

    tmin = 0.0
    tmax = 0.2
    dt = 0.01

    # encoder
    encoder_inputs = keras.Input(shape=(2,))
    x = keras.layers.Dense(512, activation="relu")(encoder_inputs)
    x = keras.layers.Dense(256, activation="tanh")(x)
    # sampling layer
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_var = keras.layers.Dense(latent_dim, name="z_var")(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_var], name="encoder")

    sampler = DiffusionSampler(paths, epochs)

    diffusion = Diffusion(encoder, paths, tmin, tmax, dt, name="diffusion-model")
    diffusion.compile(optimizer=keras.optimizers.Adam())
    diffusion.fit(
        sampler,
        epochs=epochs,
        batch_size=1,
        workers=-1,
    )


if __name__ == "__main__":
    main()
