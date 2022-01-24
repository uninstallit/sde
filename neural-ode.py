import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")
import tensorflow as tf

# pip3 install --upgrade git+https://github.com/titu1994/tfdiffeq.git
from tfdiffeq import odeint
from tfdiffeq import plot_phase_portrait, plot_vector_field, plot_results


class Lambda(tf.keras.Model):

    mu = tf.constant(0.05, dtype=tf.double)
    sigma = tf.constant(0.5, dtype=tf.double)

    def call(self, t, x):

        dx = self.mu * 0.01 + self.sigma * tf.cast(
            tf.math.sqrt(0.001), dtype=tf.double
        ) * tf.random.normal([784], dtype=tf.double)
        x = x + dx

        return x


def main():

    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    image = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[1]))[0]

    NUM_SAMPLES = 1000
    initial_states = tf.convert_to_tensor(image, dtype=tf.float64)
    t = tf.linspace(0.0, 1.0, num=NUM_SAMPLES)

    result = odeint(Lambda(), initial_states, t)

    plot_results(t, result)
    plt.show()


if __name__ == "__main__":
    main()
