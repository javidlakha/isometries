import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import scipy.io as sio
from jax import random
from tqdm import tqdm


def apply_noise(img, key, noise_level):
    noise = random.normal(key, img.shape) * noise_level
    noisy_img = jnp.clip(img + noise, 0.0, 1.0)
    return noisy_img


def apply_noise_batch(x, key, noise_level):
    keys = random.split(key, x.shape[0])
    return jax.vmap(lambda img, k: apply_noise(img, k, noise_level))(x, keys)


def load_padded_mnist(num_digits):
    ds = tfds.load('mnist', split='test', batch_size=num_digits, as_supervised=True)
    images, labels = next(iter(tfds.as_numpy(ds)))

    images = images.astype(np.float32) / 255.0
    padded = np.zeros((images.shape[0], 40, 40, 1), dtype=np.float32)
    padded[:, 6:34, 6:34, :] = images

    return padded, labels


def generate_noisy_dataset(images, labels, num_samples, noise_level):
    num_digits = images.shape[0]
    noisy_all = []
    key = random.PRNGKey(0)

    for i in tqdm(range(num_samples), desc="Applying Gaussian noise"):
        key, subkey = random.split(key)
        noisy = apply_noise_batch(jnp.array(images), subkey, noise_level=noise_level)
        noisy_np = np.array(noisy).astype(np.float32)
        noisy_all.append(noisy_np)

    noisy_all = np.stack(noisy_all, axis=1)
    return noisy_all, labels


def export_to_mat(noisy_imgs, labels, filename):
    N, M, H, W, C = noisy_imgs.shape
    total = N * M

    img_data = noisy_imgs.reshape((total, H, W)).astype(np.float32)
    labels_arr = np.repeat(labels, M).astype(np.float32)

    sio.savemat(filename, {
        'img_data': img_data,
        'labels': labels_arr
    }, oned_as='column')


def visualize_noisy_batch(noisy_imgs, num_digits, num_show, path):
    fig, axs = plt.subplots(num_digits, num_show, figsize=(num_show * 2, num_digits * 2))
    for i in range(num_digits):
        for j in range(num_show):
            axs[i, j].imshow(noisy_imgs[i, j, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize Gaussian-noised MNIST images.")
    parser.add_argument('--out', type=Path, required=True, help='Output directory for saved files')
    parser.add_argument('--noise-level', type=float, required=True, help='Standard deviation of Gaussian noise')
    args = parser.parse_args()

    output_directory: Path = args.out
    noise_level = args.noise_level

    output_directory.mkdir(parents=True, exist_ok=True)

    images, labels = load_padded_mnist(num_digits=10_000)
    noisy_imgs, labels = generate_noisy_dataset(images, labels, num_samples=32, noise_level=noise_level)
    print(f"Noisy images shape: {noisy_imgs.shape}")

    export_to_mat(noisy_imgs, labels, filename=output_directory / 'dataset.mat')
    visualize_noisy_batch(noisy_imgs, num_digits=5, num_show=6, path=output_directory / "dataset.png")
