import numpy as np
import matplotlib.pyplot as plt


def plot_single_img(img, title, size=(10, 10)):
    """Plot a single image"""
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_n_imgs(imgs, n_rows, n_cols, titles=None, size=(30, 30)):
    """Plot n images"""
    plt.figure(figsize=size)
    for i in range(len(imgs)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()
