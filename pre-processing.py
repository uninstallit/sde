import numpy as np
import tensorflow as tf
from scipy.stats import rv_continuous, norm

class GaussianProduct(rv_continuous):
    def __init__(self, means, stds, a, b, **kwargs):
        super(GaussianProduct, self).__init__(**kwargs)
        self.means = means
        self.stds = stds
        self.a = a
        self.b = b

    def gaussian(self, x, mu, sigma): 
        return norm.pdf(x, loc=mu, scale=sigma)

    def _pdf(self, x):
        pdf = 1.0
        for mu, sigma in zip(self.means, self.stds):
            pdf = pdf * norm.pdf(x, loc=mu, scale=sigma)
            # print("mu: {} - sigma: {} - x: {:.3f} - pdf: {}".format(mu, sigma, x, norm.pdf(x, loc=mu, scale=sigma)))
        return pdf

    # def _get_support(self):
    #     return [self.a, self.b]


def main():

    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[1]))
    
    class_means = []
    class_stds = []
    for cls in range(10):
        indices = np.where(labels==cls)
        class_images = np.take(images, indices, axis=0)[0]

        means = []
        stds = []
        weights = []
        for index in range(images.shape[1]):
            px = class_images[:,index]
            weight = px.shape[0] / images.shape[0]
            weights.append(weight)
            means.append(np.mean(px))
            stds.append(np.std(px) + 1000.0)

        class_means.append(means)
        class_stds.append(stds)


    for cls in range(10):
        cls_means = np.array(class_means[cls]).astype(np.float32)
        cls_stds = np.array(class_stds[cls]).astype(np.float32)
        joint_gaussian = GaussianProduct(means=cls_means, stds=cls_stds, a=0.0, b=255.0, name='joint-indep.-gaussian')

        print("class: {} - mean: {}: ".format(cls, joint_gaussian.mean()))
        print("class: {} - std:  {}: ".format(cls, joint_gaussian.std()))
    
    print("mixed gaussian weights: {:.3f}".format(weights))


if __name__ == '__main__':
    main()
