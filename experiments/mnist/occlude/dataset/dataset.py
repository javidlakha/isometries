import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm


def make_rotated_square_mask(H, W, box_size, angle_deg, cx, cy):
  angle = jnp.deg2rad(angle_deg)
  s = box_size / 2.0
  corners = jnp.array([[-s, -s], [s, -s], [s, s], [-s, s]])
  rot = jnp.array([
    [jnp.cos(angle), -jnp.sin(angle)],
    [jnp.sin(angle),  jnp.cos(angle)]
  ])
  rotated_corners = (rot @ corners.T).T + jnp.array([cx, cy])
  y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
  pts = jnp.stack([x, y], axis=-1)

  def edge_fn(p0, p1, pt):
    edge = p1 - p0
    v = pt - p0
    return edge[0] * v[..., 1] - edge[1] * v[..., 0]

  inside = jnp.ones((H, W), dtype=bool)
  for i in range(4):
    p0 = rotated_corners[i]
    p1 = rotated_corners[(i + 1) % 4]
    inside &= (edge_fn(p0, p1, pts) >= 0)

  return inside.astype(jnp.float32)


def apply_rotated_occlusion(img, key, box_frac):
  H, W, C = img.shape
  key1, key2, key3 = random.split(key, 3)

  box_size = int(H * box_frac)
  cx = random.uniform(key1, (), minval=box_size // 2, maxval=H - box_size // 2)
  cy = random.uniform(key2, (), minval=box_size // 2, maxval=W - box_size // 2)
  angle = random.uniform(key3, (), minval=0.0, maxval=180.0)

  mask = make_rotated_square_mask(H, W, box_size, angle, cx, cy)[..., None]  # (H, W, 1)
  return jnp.where(mask == 1.0, 0.0, img)  # Black square occlusion


def apply_rotated_occlusion_batch(x, key, box_frac):
  keys = random.split(key, x.shape[0])
  return jax.vmap(lambda img, k: apply_rotated_occlusion(img, k, box_frac))(x, keys)


def load_padded_mnist(num_digits):
  ds = tfds.load('mnist', split='test', batch_size=num_digits, as_supervised=True)
  images, labels = next(iter(tfds.as_numpy(ds)))

  images = images.astype(np.float32) / 255.0
  padded = np.zeros((images.shape[0], 40, 40, 1), dtype=np.float32)
  padded[:, 6:34, 6:34, :] = images

  return padded, labels


def generate_occluded_dataset(images, labels, num_samples, box_frac):
  occluded_all = []
  key = random.PRNGKey(0)

  for i in tqdm(range(num_samples), desc="Applying rotated square occlusions"):
    key, subkey = random.split(key)
    occluded = apply_rotated_occlusion_batch(jnp.array(images), subkey, box_frac=box_frac)
    occluded_all.append(np.array(occluded).astype(np.float32))

  occluded_all = np.stack(occluded_all, axis=1)
  return occluded_all, labels


def export_to_mat(occluded_imgs, labels, filename):
  N, M, H, W, C = occluded_imgs.shape
  total = N * M

  img_data = occluded_imgs.reshape((total, H, W)).astype(np.float32)
  labels_arr = np.repeat(labels, M).astype(np.float32)

  sio.savemat(filename, {
    'img_data': img_data,
    'labels': labels_arr
  }, oned_as='column')


def visualize_occlusion_batch(occluded_imgs, num_digits, num_show, path):
  fig, axs = plt.subplots(num_digits, num_show, figsize=(num_show * 2, num_digits * 2))
  for i in range(num_digits):
    for j in range(num_show):
      axs[i, j].imshow(occluded_imgs[i, j, :, :, 0], cmap='gray')
      axs[i, j].axis('off')
  plt.savefig(path)
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate and visualize rotated-occluded MNIST images.")
  parser.add_argument('--out', type=Path, required=True, help='Output directory for saved files')
  parser.add_argument('--box-frac', type=float, required=True, help='Box size as fraction of image size (e.g., 0.5)')
  args = parser.parse_args()

  output_directory: Path = args.out
  box_frac = args.box_frac

  output_directory.mkdir(parents=True, exist_ok=True)

  images, labels = load_padded_mnist(num_digits=10_000)
  occluded_imgs, labels = generate_occluded_dataset(images, labels, num_samples=32, box_frac=box_frac)
  print(f"Rotated-occluded images shape: {occluded_imgs.shape}")

  export_to_mat(occluded_imgs, labels, filename=output_directory / 'dataset.mat')
  visualize_occlusion_batch(occluded_imgs, num_digits=5, num_show=6, path=output_directory / "dataset.png")
