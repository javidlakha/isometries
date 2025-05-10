import os
import argparse
import numpy as np
import jax.numpy as jnp
import potpourri3d as pp3d
import scipy.sparse
import scipy.sparse.linalg as sla
import tensorflow as tf
import trimesh
from tqdm import tqdm
from scipy.sparse.linalg import spsolve

from data import data_loader as dl
from utils import ioutils as io


# Constants
NUM_XFORMS = 30
ALPHA = 10
SMOOTHNESS = 0.3
NOISE_LEVEL = 0.05
DROPOUT_RATIO = 0.5
AMPUTATION_PERCENTILE = 75
NUM_EIGS = 128
NUM_CHANNELS = 16
S2_RES = (96, 192)
NUM_CLASS_TRAIN = 10
NUM_CLASS_TEST = 4
SKIP = {"glasses", "lamp", "snake", "two_balls", "myScissor"}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--transform", required=True)
parser.add_argument("--cls", required=True)
args = parser.parse_args()

transform = args.transform
cls = args.cls

# Directories
BASE_ROOT = "data/shrec11_conf"
BASE_DIR = os.path.join(BASE_ROOT, "base")
AUG_ROOT = "data/shrec11_aug/"
AUG_DIR = os.path.join(AUG_ROOT, f"{transform}_aug")
SPHERE_DIR = os.path.join(AUG_ROOT, f"{transform}_sphere")
OUT_DIR = f"data/shrec11_aug/processed_{transform}"
TRAIN_DIR = os.path.join(OUT_DIR, "train")
TEST_DIR = os.path.join(OUT_DIR, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

if cls in SKIP:
    print(f"Skipping {cls}")
    exit()

# Sphere projection
def naive_sphere_projection(V):
    V_centered = V - V.mean(axis=0)
    norms = np.linalg.norm(V_centered, axis=1, keepdims=True)
    return V_centered / (norms + 1e-8)

def save_deformed_mesh(V, F, out_dir, mesh_id, i):
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.export(os.path.join(out_dir, f"{i}.ply"))

def save_sphere_projection(V, F, out_dir, mesh_id, i):
    V_sphere = naive_sphere_projection(V)
    save_deformed_mesh(V_sphere, F, out_dir, mesh_id, i)

# Deformations
def elastic_deform(V, F, alpha, smoothness, seed):
    np.random.seed(seed)
    N = V.shape[0]
    displacement = np.random.randn(N, 3)
    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    M = pp3d.vertex_areas(V, F)
    M += 1e-8 * np.mean(M)
    Mmat = scipy.sparse.diags(M)
    A = Mmat + smoothness * L
    smoothed_disp = np.zeros_like(displacement)
    for d in range(3):
        rhs = M * displacement[:, d]
        smoothed_disp[:, d] = spsolve(A, rhs)
    return V + alpha * smoothed_disp, F

def dropout_deform(V, F, ratio, seed):
    np.random.seed(seed)
    keep_mask = np.random.rand(V.shape[0]) > ratio
    keep_indices = np.where(keep_mask)[0]
    index_map = -np.ones(V.shape[0], dtype=int)
    index_map[keep_indices] = np.arange(len(keep_indices))
    F_new = F[np.all(keep_mask[F], axis=1)]
    F_new = index_map[F_new]
    V_new = V[keep_indices]
    mesh = trimesh.Trimesh(vertices=V_new, faces=F_new, process=False)
    trimesh.repair.fill_holes(mesh)
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh.vertices, mesh.faces

def noise_deform(V, F, noise_level, seed):
    np.random.seed(seed)
    return V + np.random.normal(scale=noise_level, size=V.shape), F

def amputation_deform(V, F, amputation_percentile):
    z_vals = V[:, 2]
    z_thresh = np.percentile(z_vals, amputation_percentile)
    keep = z_vals > z_thresh
    index_map = -np.ones(V.shape[0], dtype=int)
    keep_indices = np.where(keep)[0]
    index_map[keep_indices] = np.arange(len(keep_indices))
    F_new = F[np.all(keep[F], axis=1)]
    F_new = index_map[F_new]
    V_new = V[keep]
    return V_new, F_new

def get_spectrum(V, F, k):
    EPS = 1.0e-8
    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    mvec = pp3d.vertex_areas(V, F)
    mvec += EPS * np.mean(mvec)
    L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * EPS).tocsc()
    Mmat = scipy.sparse.diags(mvec)
    failcount = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=EPS)
            evals = np.clip(evals, 0., float('inf'))
            return evecs, evals, mvec
        except:
            if failcount > 3:
                raise
            failcount += 1
            L_eigsh += scipy.sparse.identity(L.shape[0]) * (EPS * 10**failcount)

# Sphere sampling grid
I, J = np.meshgrid(np.arange(S2_RES[0]), np.arange(S2_RES[1]), indexing="ij")
I = (I / (S2_RES[0] - 1)) * np.pi
J = (J / (S2_RES[1] - 1)) * 2.0 * np.pi
X = np.cos(J) * np.sin(I)
Y = np.sin(J) * np.sin(I)
Z = np.cos(I)
P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

# Class-level processing
in_class_dir = os.path.join(BASE_DIR, cls)
mesh_files = sorted([f for f in os.listdir(in_class_dir) if f.endswith(".ply")])
mesh_files = mesh_files[:NUM_CLASS_TRAIN + NUM_CLASS_TEST]

train_writer = tf.io.TFRecordWriter(os.path.join(TRAIN_DIR, f"{cls}.tfrecords"))
test_writer = tf.io.TFRecordWriter(os.path.join(TEST_DIR, f"{cls}.tfrecords"))

saved_keys = False
MEAN = 0
MEAN2 = 0
stat_count = 0
label = 0

for j, mf in enumerate(mesh_files):
    mesh_id = os.path.splitext(mf)[0]
    in_path = os.path.join(in_class_dir, mf)
    mesh = trimesh.load(in_path, process=False)
    V, F = mesh.vertices, mesh.faces

    for i in tqdm(range(NUM_XFORMS), desc=f"Augment {transform} {cls}/{mesh_id}"):
        success = False
        for attempt in range(5):
            try:
                seed = i + attempt * 1000
                if transform == "elastic":
                    V_def, F_def = elastic_deform(V, F, ALPHA, SMOOTHNESS, seed)
                elif transform == "dropout":
                    V_def, F_def = dropout_deform(V.copy(), F.copy(), DROPOUT_RATIO, seed)
                elif transform == "noisy":
                    V_def, F_def = noise_deform(V.copy(), F.copy(), NOISE_LEVEL, seed)
                elif transform == "amputation":
                    V_def, F_def = amputation_deform(V, F, AMPUTATION_PERCENTILE)
                else:
                    raise ValueError(f"Unknown transform: {transform}")

                if len(V_def) == 0 or len(F_def) == 0:
                    raise ValueError("Empty mesh")

                mesh_out_dir = os.path.join(AUG_DIR, cls, mesh_id)
                sphere_out_dir = os.path.join(SPHERE_DIR, cls, mesh_id)
                save_deformed_mesh(V_def, F_def, mesh_out_dir, mesh_id, i)
                save_sphere_projection(V_def, F_def, sphere_out_dir, mesh_id, i)
                success = True
                break
            except Exception as e:
                print(f"Retry {attempt + 1}/5 failed for {cls}/{mesh_id}/{i}: {e}")

        if not success:
            print(f"Skipping {cls}/{mesh_id} transform {i} after 5 failed attempts")

    # HKS preprocessing
    sphere_path = os.path.join(SPHERE_DIR, cls, mesh_id, "0.ply")
    V_base, F_base = io.load_mesh(sphere_path)
    if V_base is None or F_base is None:
        print(f"Failed to load base mesh: {sphere_path}")
        continue

    Phi, Lambda, Mass = get_spectrum(V_base, F_base, NUM_EIGS)
    t_vals = -1.0 * np.logspace(-2.0, 0.0, num=NUM_CHANNELS)
    signal = np.sum((Phi[..., None]**2) * np.exp(Lambda[None, :, None] * t_vals[None, None, :]), axis=-2)

    s_mean = np.sum(signal * Mass[:, None], axis=0) / np.sum(Mass)
    s_mean2 = np.sum((signal**2) * Mass[:, None], axis=0) / np.sum(Mass)
    MEAN += s_mean
    MEAN2 += s_mean2
    stat_count += 1

    mesh0 = trimesh.Trimesh(vertices=V_base, faces=F_base)
    points0, _, tID0 = trimesh.proximity.closest_point(mesh0, P)
    bary0 = trimesh.triangles.points_to_barycentric(V_base[F_base[tID0]], points0, method='cross')
    base_signal = jnp.sum(signal[F_base[tID0]] * bary0[..., None], axis=1)
    base_signal = np.reshape(base_signal, (S2_RES[0], S2_RES[1], NUM_CHANNELS))

    maps = np.zeros((NUM_XFORMS + 1, S2_RES[0], S2_RES[1], NUM_CHANNELS), dtype=np.float32)
    maps[0] = base_signal

    for l in range(NUM_XFORMS):
        sphere_path = os.path.join(SPHERE_DIR, cls, mesh_id, f"{l}.ply")
        V_def, F_def = io.load_mesh(sphere_path)
        if V_def is None or F_def is None or len(V_def) == 0 or len(F_def) == 0:
            continue

        try:
            mesh_def = trimesh.Trimesh(vertices=V_def, faces=F_def)
            pointsM, _, tIDM = trimesh.proximity.closest_point(mesh_def, P)
            face_indices = F_def[tIDM]

            if face_indices.max() >= signal.shape[0]:
                print(f"Skipping {cls}/{mesh_id} transform {l}: face index {face_indices.max()} exceeds signal length {signal.shape[0]}")
                continue

            baryM = trimesh.triangles.points_to_barycentric(V_def[face_indices], pointsM, method='cross')
            mob_signal = jnp.sum(signal[face_indices] * baryM[..., None], axis=1)
            mob_signal = np.reshape(mob_signal, (S2_RES[0], S2_RES[1], NUM_CHANNELS))
            maps[l+1] = mob_signal
        except Exception as e:
            print(f"Projection failed for {cls}/{mesh_id} transform {l}: {e}")
            continue

    geom = {
        "label": label * np.ones((NUM_XFORMS + 1,), dtype=np.int32),
        "image": maps,
    }

    ex, shape_keys = dl.encode_example(geom)
    if not saved_keys:
        io.save_dict(os.path.join(TRAIN_DIR, "shape_keys.json"), shape_keys)
        io.save_dict(os.path.join(TEST_DIR, "shape_keys.json"), shape_keys)
        saved_keys = True

    if j < NUM_CLASS_TRAIN:
        train_writer.write(ex)
    else:
        test_writer.write(ex)

train_writer.close()
test_writer.close()

if stat_count > 0:
    MEAN /= stat_count
    MEAN2 /= stat_count
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"stats_{cls}.npz"), MEAN=MEAN, MEAN2=MEAN2)

print("Done.")
