import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import scipy.io as sio
from jax import random, lax
from tqdm import tqdm


def gaussian_kernel1d(sigma, radius=3):
  x = jnp.arange(-radius, radius + 1)
  kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
  return kernel / jnp.sum(kernel)


def gaussian_blur(img, sigma):
  kernel = gaussian_kernel1d(sigma)
  img = img[None, ..., None]  # (1, H, W, 1)

  blur_v = lax.conv_general_dilated(
    img,
    kernel[:, None, None, None],
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC")
  )

  blur_h = lax.conv_general_dilated(
    blur_v,
    kernel[None, :, None, None],
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC")
  )

  return blur_h[0, ..., 0]


def apply_elastic_deformation(img, key, alpha, sigma):
  H, W, C = img.shape
  key_dx, key_dy = random.split(key)

  img = jnp.clip(img, 0.0, 1.0).astype(jnp.float32)

  dx = random.uniform(key_dx, (H, W), minval=-1.0, maxval=1.0)
  dy = random.uniform(key_dy, (H, W), minval=-1.0, maxval=1.0)

  dx = gaussian_blur(dx, sigma=sigma) * alpha
  dy = gaussian_blur(dy, sigma=sigma) * alpha

  x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
  indices_x = jnp.clip(x + dx, 0, W - 1)
  indices_y = jnp.clip(y + dy, 0, H - 1)

  def bilinear_sample(img, x, y):
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    def get_val(ix, iy):
      return img[iy, ix, :]

    Ia = get_val(x0, y0)
    Ib = get_val(x0, y1)
    Ic = get_val(x1, y0)
    Id = get_val(x1, y1)

    return (
      wa[..., None] * Ia +
      wb[..., None] * Ib +
      wc[..., None] * Ic +
      wd[..., None] * Id
    )

  return bilinear_sample(img, indices_x, indices_y)


def apply_elastic_batch(x, key, alpha, sigma, batch_size=1000):
  n = x.shape[0]
  keys = random.split(key, n)

  def process_chunk(start):
    end = jnp.minimum(start + batch_size, n)
    x_chunk = x[start:end]
    key_chunk = keys[start:end]
    return jax.vmap(lambda img, k: apply_elastic_deformation(img, k, alpha, sigma))(x_chunk, key_chunk)

  chunks = [process_chunk(i) for i in range(0, n, batch_size)]
  return jnp.concatenate(chunks, axis=0)


def load_padded_mnist(num_digits):
  ds = tfds.load('mnist', split='test', batch_size=num_digits, as_supervised=True)
  images, labels = next(iter(tfds.as_numpy(ds)))

  images = images.astype(np.float32) / 255.0
  padded = np.zeros((images.shape[0], 40, 40, 1), dtype=np.float32)
  padded[:, 6:34, 6:34, :] = images

  return padded, labels


def generate_elastic_dataset(images, labels, num_samples, alpha, sigma):
  warped_all = []
  key = random.PRNGKey(0)

  for i in tqdm(range(num_samples), desc="Applying elastic deformation"):
    key, subkey = random.split(key)
    warped = apply_elastic_batch(jnp.array(images), subkey, alpha=alpha, sigma=sigma, batch_size=25)
    warped_np = np.array(warped).astype(np.float32)
    warped_all.append(warped_np)

  warped_all = np.stack(warped_all, axis=1)
  return warped_all, labels


def export_to_mat(elastic_imgs, labels, filename):
  N, M, H, W, C = elastic_imgs.shape
  total = N * M

  img_data = elastic_imgs.reshape((total, H, W)).astype(np.float32)
  labels_arr = np.repeat(labels, M).astype(np.float32)

  sio.savemat(filename, {
    'img_data': img_data,
    'labels': labels_arr
  }, oned_as='column')


def visualize_elastic_batch(elastic_imgs, num_digits, num_show, path):
  fig, axs = plt.subplots(num_digits, num_show, figsize=(num_show * 2, num_digits * 2))
  for i in range(num_digits):
    for j in range(num_show):
      axs[i, j].imshow(elastic_imgs[i, j, :, :, 0], cmap='gray')
      axs[i, j].axis('off')
  plt.savefig(path)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate and visualize elastic-deformed MNIST images.")
  parser.add_argument('--out', type=Path, required=True, help='Output directory for saved files')
  parser.add_argument('--alpha', type=float, required=True, help='Elastic deformation intensity')
  parser.add_argument('--sigma', type=float, required=True, help='Elastic deformation smoothing')
  args = parser.parse_args()

  output_directory: Path = args.out
  alpha = args.alpha
  sigma = args.sigma

  output_directory.mkdir(parents=True, exist_ok=True)

  images, labels = load_padded_mnist(num_digits=10_000)
  elastic_imgs, labels = generate_elastic_dataset(images, labels, num_samples=32, alpha=alpha, sigma=sigma)
  print(f"Elastic-deformed images shape: {elastic_imgs.shape}")

  export_to_mat(elastic_imgs, labels, filename=output_directory / 'dataset.mat')
  visualize_elastic_batch(elastic_imgs, num_digits=5, num_show=6, path=output_directory / "dataset.png")
