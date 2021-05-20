import os
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
os.environ["CUDA_VISIBLE_DEVICES"] = " "
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# let an nn ~ p(x,t)
# let diffusion dx(t) = mu dt + sig dW(t)
# dp(x,t)/dt = -mu dp(x,t)/dx + sig^2/2 d^2p(x,t)/dx^2


class ODENetwork():

    def __init__(self):
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.history = []

        self.xmin = -5.0
        self.xmax = 5.0
        self.tmin = 0.0
        self.tmax = 5.0

        self.mu = tf.constant(0.05, dtype=tf.double)
        self.sig2 = tf.constant(0.125, dtype=tf.double)

        self.normal_dist = tfd.Normal(loc=0.0, scale=0.01)

    def gaussian(self, x):
        mu = tf.cast(0.0, dtype=tf.double)
        sigma = tf.cast(1.0, dtype=tf.double)
        Z = tf.cast((2.0 * np.pi * keras.backend.square(sigma))
                    ** 0.5, dtype=tf.double)
        return tf.cast(keras.backend.exp(-0.5 * (x - mu)**2.0 / sigma**2.0) / Z, dtype=tf.double)

    def get_data(self):
        x = np.linspace(self.xmin, self.xmax, num=1000)
        t = np.linspace(self.tmin, self.tmax, num=1000)
        xx, tt = np.meshgrid(x, t)
        x_train = np.column_stack((xx.ravel(), tt.ravel()))
        return x, t, xx, tt, x_train

    def create_model(self):
        inputs = keras.Input(shape=(2, ))
        x = layers.Dense(100, activation="sigmoid")(inputs)  # self.gaussian
        x = layers.Dense(100, activation="sigmoid")(x)
        x = layers.Dense(100, activation="sigmoid")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        self.model = keras.Model(
            inputs=inputs, outputs=outputs, name="experimental")
        print(self.model.summary())
        return self

    @tf.function
    def loss(self, point, p, dp_dt, dp_dx, d2p_dx2):
        point = tf.cast(point, dtype=tf.double)

        # x = tf.cast(point[0], dtype=tf.float32)
        # ic = tf.cast(self.normal_dist.prob(x) / 0.01, dtype=tf.double)

        ic = self.gaussian(point[0])

        # initial  t0
        point_xt0 = tf.expand_dims(tf.math.multiply(
            point,
            tf.constant(np.array([1, 0]), dtype=tf.double)), axis=0)
        p_xtmin = tf.cast(self.model(point_xt0, training=False),
                          dtype=tf.double)

        # xmin
        point_x = tf.math.multiply(
            point,
            tf.constant(np.array([0, 1]), dtype=tf.double))
        point_xmin = tf.math.add(
            point_x,
            tf.constant(np.array([self.xmin, 0]), dtype=tf.double))
        point_xmin = tf.expand_dims(point_xmin, axis=0)
        p_xmint = tf.cast(self.model(
            point_xmin, training=False), dtype=tf.double)

        # xmax
        point_xmax = tf.math.add(
            point_x,
            tf.constant(np.array([self.xmax, 0]), dtype=tf.double))
        point_xmax = tf.expand_dims(point_xmax, axis=0)
        p_xmaxt = tf.cast(self.model(
            point_xmax, training=False), dtype=tf.double)

        loss = 100.0 * tf.keras.backend.abs(p_xtmin - ic) +   \
            tf.keras.backend.abs(p_xmint) +           \
            tf.keras.backend.abs(p_xmaxt) +           \
            tf.keras.backend.abs(dp_dt + self.mu * dp_dx - self.sig2 * d2p_dx2)
        # tf.keras.backend.abs(
        #     tf.keras.backend.maximum(p, 0.0))
        return loss

    @tf.function
    def apply_training_step(self, point):

        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(point)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(point)

                # function value
                u = tf.cast(self.model(point, training=True), dtype=tf.double)

                # first derivatives
                grads = tape_ord_1.gradient(u, point)
                du_dx = grads[:, 0]
                du_dt = grads[:, 1]

                # second derivatives
                grads2x = tape_ord_2.gradient(du_dx, point)
                d2u_dx2 = grads2x[:, 0]

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss(x[0], x[1], x[2], x[3], x[4]),
                    (point, u, du_dt, du_dx, d2u_dx2),
                    dtype=tf.double)
                loss = tf.reduce_mean(loss, axis=-1)

        grads = tape_ord_1.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        return loss

    def train(self):
        self.epochs = 1
        self.batch_size = 128

        x, t, xx, tt, x_train = self.get_data()
        xt_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        xt_dataset = xt_dataset.shuffle(
            buffer_size=1024, seed=1234).batch(self.batch_size)

        print("dataset length: ", x_train.shape)

        for epoch in range(self.epochs):

            for step, point in enumerate(xt_dataset):
                loss = self.apply_training_step(point)

                if step % 200 == 0:
                    self.history.append(tf.reduce_mean(loss))
                    print("Training loss for step/epoch {}/{}: {}".format(step,
                                                                          epoch, tf.reduce_mean(loss)))

    def predict(self):
        x, t, xx, tt, x_train = self.get_data()
        predictions = self.model.predict(x_train)
        return predictions

    def get_history(self):
        return self.history


def main():

    ode_net = ODENetwork()
    ode_net.create_model()
    ode_net.train()

    plt.plot(ode_net.get_history())
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    x, t, xx, tt, x_train = ode_net.get_data()
    z = ode_net.predict().reshape((t.shape[0], x.shape[0]))
    # Plot the surface.
    surf = ax.plot_surface(tt, xx, z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    plt.title('ODE Solution')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.contourf(t, x, z)
    plt.show()


if __name__ == '__main__':
    main()
