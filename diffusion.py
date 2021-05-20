import numpy as np
import tensorflow as tf


def main():
    
    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    rescaled_images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[1]))

    indices = np.where(labels==0)
    zero_images = np.take(rescaled_images, indices, axis=0)[0]
    x0 = np.array(zero_images).astype(np.float32)

    length = zero_images.shape[1]
    ones = np.ones(length)
    
    steps = 1000000 # <--- problem even with one step
    mu = -2.0
    sigma = 1.5
    dt = 0.01

    data = []
    for image in x0:
        diff_imgs = [image]
        x = x0
        for i in range(steps):
            x = x + mu * ones * dt + sigma * ones * np.sqrt(dt) * np.random.randn(length)
            diff_imgs.append(x)

        with open('data.npy','wb') as f:
            data.append(diff_imgs)

    data = np.array(data, dtype=np.float32)

    # plt.imshow(np.array(x).reshape(images.shape[1], images.shape[1]))
    # plt.show(block=True)

if __name__ == '__main__':
    main()
