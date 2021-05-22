import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class DiffusionSampler(tf.keras.utils.Sequence):
    def __init__(self, x, y, mu, sigma, tmin, tmax, dt, number_images, *args, **kwargs):
        self.sample_size = x.shape[0]
        self.img_dim = x.shape[1]
        self.x = np.reshape(x, (self.sample_size, self.img_dim * self.img_dim))
        self.y = y

        self.height = self.x.shape[1]
        self.number_images = number_images

        self.mu = mu
        self.sigma = sigma
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

        self.time = np.arange(tmin, tmax, dt)
        self.max_steps = len(self.time)

    def diffuse_image(self, image, steps):
        x = image.astype(np.float32)
        for i, step in enumerate(steps):
            # each pixel
            for j in range(step):
                # time step
                x[i] = (
                    x[i]
                    + self.mu * self.dt
                    + self.sigma * np.sqrt(self.dt) * np.random.randn()
                )
        return x

    def __len__(self):
        return np.ceil(self.sample_size / self.number_images).astype(int)

    def __getitem__(self, index):
        image_idxs = np.random.randint(self.sample_size - 1, size=self.number_images)
        rand_steps = np.reshape(
            np.random.randint(
                0, self.max_steps - 1, size=(self.height * self.number_images)
            ),
            (self.number_images, self.height),
        )
        batch_imgs = self.x[image_idxs]

        batch_x0 = []
        for image in batch_imgs:
            batch_x0.append([image, np.zeros(len(image))])
        batch_x0 = np.array(batch_x0)
        batch_x0 = np.squeeze(batch_x0)

        batch_x = []
        for image, steps in zip(batch_imgs, rand_steps):
            x = self.diffuse_image(image, steps)
            t = self.time[steps]
            batch_x.append([x, t])
        batch_x = np.array(batch_x)
        batch_x = np.squeeze(batch_x)

        batch_y = self.y[image_idxs]
        batch_x = np.hstack((batch_x0, batch_x))
        return batch_x, batch_y

    # def on_epoch_end(self):
    # option method to run some logic at the end of each epoch: e.g. reshuffling


class MonitorCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):

        if batch % 10 == 0:

            # Get data.
            x, t, xx, tt, x_train = self.model.points
            z = self.model.predict(x_train).reshape((t.shape[0], x.shape[0]))

            # Plot the surface.
            self.model.ax.clear()
            surf = self.model.ax.plot_surface(
                tt, xx, z, cmap=cm.viridis, linewidth=0, antialiased=False
            )

            # Customize the z axis.
            self.model.ax.set_zlim(-0.01, 0.5)
            self.model.ax.zaxis.set_major_locator(LinearLocator(10))
            self.model.ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

            self.model.ax.set_xlabel("x")
            self.model.ax.set_xlabel("t")

            plt.pause(0.01)


class TroubleshootingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):

        batch_x0 = logs["batch"]

        # keys = list(logs.keys())
        # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))


class ForwardKolmogorovSolver(tf.keras.Model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    initializer = tf.keras.initializers.GlorotUniform()

    zero = tf.constant(0.0, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    two = tf.constant(2.0, dtype=tf.float32)
    pi = tf.constant(np.pi, dtype=tf.float32)

    loss_f_trckr = tf.keras.metrics.Mean(name="loss_f")

    hist = {
        "loss_f": [],
    }

    x = np.linspace(-255, 255, num=1000)
    t = np.linspace(0, 10, num=1000)
    xx, tt = np.meshgrid(x, t)
    x_train = np.column_stack((xx.ravel(), tt.ravel()))
    points = (x, t, xx, tt, x_train)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plt.show(block=False)

    # @tf.function
    def loss_f(self, point, p_xt, dp_dt, dp_dx, d2p_dx2):

        loss_f = keras.backend.abs(dp_dt + self.mu * dp_dx - self.sigma * d2p_dx2)
        return loss_f

    @tf.function
    def loss_ic(self, point):
        loss_ic = 1.0
        return loss_ic

    def train_step(self, data):
        batch_x, batch_y = data

        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(batch_x)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(batch_x)

                batch_x = tf.transpose(batch_x)
                p_xt = self(batch_x, training=True)

                # # first derivatives
                grads = tape_ord_1.gradient(p_xt, batch_x)
                du_dx = grads[:, 0]
                du_dt = grads[:, 1]

                # # second derivatives
                grads2x = tape_ord_2.gradient(du_dx, batch_x)
                d2u_dx2 = grads2x[:, 0]

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss_f(x[0], x[1], x[2], x[3], x[4]),
                    (batch_x, p_xt, du_dt, du_dx, d2u_dx2),
                    dtype=tf.float32,
                )
                loss = tf.reduce_mean(loss, axis=-1)

        self.loss_f_trckr.update_state(loss)

        grads = tape_ord_1.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss_f": self.loss_f_trckr.result(),
        }

    @property
    def metrics(self):
        return [self.loss_f_trckr]

    def set_diffusion_params(self, mu, sigma):
        self.mu = tf.constant(mu, dtype=tf.float32)
        self.sigma = tf.constant((sigma ** 2.0 / 2.0), dtype=tf.float32)


def main():

    # (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    # indices = np.where(labels == 0)
    # images = np.squeeze(np.take(images, indices, axis=0))
    # labels = np.squeeze(np.take(labels, indices, axis=0))

    # # pixels = np.reshape(images, (images.shape[0] * images.shape[1] * images.shape[1]))
    # # _ = plt.hist(pixels, bins='auto', density=False)  # arguments are passed to np.histogram
    # # plt.title("Histogram with 'auto' bins")
    # # plt.show()

    # diffusion_ds = DiffusionSampler(
    #     x=images,
    #     y=labels,
    #     mu=0.0,
    #     sigma=3.5,
    #     tmin=0.0,
    #     tmax=10.0,
    #     dt=0.01,
    #     number_images=1,
    # )

    # inputs = keras.Input(shape=(2,))
    # x = keras.layers.Dense(50, activation="relu")(inputs)
    # x = keras.layers.Dense(50, activation="sigmoid")(x)
    # x = keras.layers.Dense(50, activation="sigmoid")(x)
    # outputs = keras.layers.Dense(1, activation="linear")(x)
    # model = ForwardKolmogorovSolver(inputs, outputs)
    # model.set_diffusion_params(0.0, 3.5)

    # print(model.summary())
    # model.compile()

    # epochs = 1

    # model.fit(
    #     diffusion_ds,
    #     epochs=epochs,
    #     workers=-1,
    #     verbose=1,
    #     callbacks=[MonitorCallback()], # TroubleshootingCallback()
    # )

    test = tf.exp(tf.lgamma(5.0))
    print("test: ", test)

    # batch_x0, batch_x, batch_y = diffusion_ds.__getitem__(1)

    # test = np.hstack((batch_x0, batch_x))

    # # test = np.expand_dims(batch_x0, -1)
    # print("test: ", test.shape)

    # mod = np.transpose(np.squeeze(batch_x0))

    # plt.imshow(np.array(batch_x[0][0]).reshape(images.shape[1], images.shape[1]))
    # plt.show(block=True)


if __name__ == "__main__":
    main()
