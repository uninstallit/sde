import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
                x[i] = x[i] + self.mu * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        return x

    def __len__(self):
        return np.ceil(self.sample_size / self.number_images).astype(int)

    def __getitem__(self, index):
        image_idxs = np.random.randint(self.sample_size - 1, size=self.number_images)
        rand_steps = np.reshape(np.random.randint(self.max_steps-1, size=(self.height * self.number_images)), (self.number_images, self.height))

        batch_imgs = self.x[image_idxs]
        batch_y = self.y[image_idxs]
        
        batch_x0 = []
        for image in batch_imgs:
            batch_x0.append([image, np.zeros(len(image))])
        batch_x0 = np.array(batch_x0)
  
        batch_x = []
        for image, steps in zip(batch_imgs, rand_steps):
            x = self.diffuse_image(image, steps)
            t = self.time[steps]
            batch_x.append([x, t])
        batch_x = np.array(batch_x)

        batch_y = self.y[image_idxs]
        return batch_x0, batch_x, batch_y

    # def on_epoch_end(self):
    # option method to run some logic at the end of each epoch: e.g. reshuffling


def main():

    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    diffusion_ds = DiffusionSampler(
        x=images, 
        y=labels, 
        mu=2.0, 
        sigma=2.5, 
        tmin=0.0, 
        tmax=100.0, 
        dt=0.01, 
        number_images=1)

    batch_x0, batch_x, batch_y = diffusion_ds.__getitem__(1)

    plt.imshow(np.array(batch_x[0][0]).reshape(images.shape[1], images.shape[1]))
    plt.show(block=True)

if __name__ == '__main__':
    main()
