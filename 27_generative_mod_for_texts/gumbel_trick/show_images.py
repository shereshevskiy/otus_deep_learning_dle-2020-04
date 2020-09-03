""" display a 2D manifold of the digits """
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical


def plot_epoch(ep_i, generator, M, N):
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    samples = np.random.randint(0, M, (N, n * n))
    cat_samples = to_categorical(samples, num_classes=M)
    cat_samples = np.reshape(cat_samples, (N, -1, n, n))

    for i in range(n):
        for j in range(n):
            z_sample = cat_samples[:, :, i, j]
            x_decoded = generator.predict(np.reshape(z_sample, (1, N * M)))
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig(f"images_gen_ep{ep_i}.png")
    plt.show()
