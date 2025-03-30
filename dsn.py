import argparse
from typing import List
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from dipy.io import image
from dipy.data import GradientTable
from dipy.reconst import shm


def compute_uni_distances(
        X: List[np.ndarray], 
        kd_trees: List[KDTree]
        ) -> List[np.ndarray]:
    """
    X: List[np.ndarray] (N, D)
    kd_trees: List[KDTree]
    return the distance from each point in X to the nearest point in X (excluding itself)
    """
    # return [uni_distance(x) for x in X]
    return [kd.query(x, 2)[0][:, 1] for x, kd in zip(X, kd_trees)]


def compute_dsn_pair(
        X: np.ndarray, 
        Y: np.ndarray, 
        r_x: np.ndarray, 
        r_y: np.ndarray, 
        kd_x: KDTree, 
        kd_y: KDTree
        ) -> float:
    """
    X: np.ndarray (N, D)
    Y: np.ndarray (M, D)
    r_x: np.ndarray (N, )
    r_y: np.ndarray (M, )
    return the distance from the nearest point in Y to the nearest point in X
    """
    D = X.shape[1]

    s_x = kd_y.query(X)[0]
    s_y = kd_x.query(Y)[0]

    xy_ratio = r_x / s_x
    yx_ratio = r_y / s_y

    xy_ratio = xy_ratio[np.isfinite(xy_ratio)]
    yx_ratio = yx_ratio[np.isfinite(yx_ratio)]

    xy_ratio = xy_ratio[xy_ratio != 0]
    yx_ratio = yx_ratio[yx_ratio != 0]

    N = xy_ratio.shape[0]
    M = yx_ratio.shape[0]

    kl_xy = -D / N * np.sum(np.log(xy_ratio)) + np.log(M / (N - 1))
    kl_yx = -D / M * np.sum(np.log(yx_ratio)) + np.log(N / (M - 1))

    return 1 / (1 + max(kl_xy, 0.0) + max(kl_yx, 0.0))


def compute_dsn(X: list[np.ndarray]) -> np.ndarray:
    """
    X: List[np.ndarray] (N, D)
    return the DSN of each point in X
    """
    N = len(X)
    kd_trees = [KDTree(x) for x in X]
    r = compute_uni_distances(X, kd_trees)
    ret = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            ret[i, j] = compute_dsn_pair(X[i], X[j], r[i], r[j], kd_trees[i], kd_trees[j])
            ret[j, i] = ret[i, j]
    return ret


def compute_dsn_from_img(img: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    X: np.ndarray (N, D)
    seg: np.ndarray (N, )
    """
    seg_vals = np.unique(seg)
    X = [img[seg == idx] for idx in seg_vals]
    return compute_dsn(X)


def compute_dsn_from_file(
        dwi_file: str, 
        bval_file: str, 
        bvec_file: str, 
        seg_file: str,
        seg_indices: List[int],
        mask_file: str = None
        ) -> np.ndarray:
    """
    dwi_file: str
    bval_file: str
    bvec_file: str
    seg_file: str
    mask_file: str
    seg_indices: List[int] indices of the segmentations to keep
    """
    dwi, _ = image.load_nifti(dwi_file)
    seg, _ = image.load_nifti(seg_file)
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file)
    bvals = np.rint(bvals / 100) * 100  # round to nearest 100
    bvals = np.clip(bvals, 0, None)

    if bvecs.shape[1] != 3:
        bvecs = bvecs.T

    seg_mask = np.isin(seg, seg_indices)
    if mask_file is not None:
        mask, _ = image.load_nifti(mask_file)
        mask = mask.astype(bool)
        seg_mask = seg_mask & mask

    seg = seg[seg_mask].astype(int)
    dwi = dwi[seg_mask, :].astype(float)    
    b0_mask = (bvals == 0)

    bval_shells = np.unique(bvals)
    bval_shells = bval_shells[bval_shells != 0]
    arr = []
    for bval in bval_shells:
        shell_mask = (bvals == bval)
        print(f"bval: {bval} shape: {dwi[:, shell_mask].shape}")
        norm_shell = np.clip(np.clip(dwi[:, shell_mask], 1e-6, None) / np.clip(np.mean(dwi[:, b0_mask], axis=1), 1.0, None)[:, None], 1e-6, 1.0)
        shell_bvecs = bvecs[shell_mask, :]
        sphm_model = shm.QballModel(GradientTable(shell_bvecs * bval), 8, assume_normed=True)
        L = sphm_model.n * (sphm_model.n + 1)
        L **= 2
        _fit_matrix = np.linalg.pinv(sphm_model.B.T @ sphm_model.B + np.diag(6e-3 * L)) @ sphm_model.B.T
        sphm_coeff = np.dot(norm_shell, _fit_matrix.T)

        sphm_rish = np.concatenate([np.square(sphm_coeff[:, 0])[:, None] / (4 * np.pi),
                                    np.sum(np.square(sphm_coeff[:, 1:6]), axis=1)[:, None] / (20 * np.pi),
                                    np.sum(np.square(sphm_coeff[:, 6:15]), axis=1)[:, None] / (36 * np.pi)], axis=1)    
        sphm_rish = np.clip(np.sqrt(sphm_rish), 0.0, 1.0)
        print(f"bval: {bval} RISH shape: {sphm_rish.shape} min: {np.min(sphm_rish)} max: {np.max(sphm_rish)}")
        arr.append(sphm_rish)
    
    arr = np.concatenate(arr, axis=1)
    return compute_dsn_from_img(arr, seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dwi_path", type=str, required=True, help="Path to the DWI data (.nii or .nii.gz)")
    parser.add_argument("--bval_path", type=str, required=True, help="Path to the bval data (.txt)")
    parser.add_argument("--bvec_path", type=str, required=True, help="Path to the bvec data (.txt)")
    parser.add_argument("--seg_path", type=str, required=True, help="Path to the Segmentation (.nii or .nii.gz)")
    parser.add_argument("--mask_path", type=str, required=False, help="Path to the Mask (.nii or .nii.gz)")
    parser.add_argument("--annot_path", type=str, required=False, default="DSN/synthseg_labels.json", help="Path to the Annotation (.json)")
    parser.add_argument("--output_path", type=str, required=False, help="Path to the output file")
    args = parser.parse_args()

    dwi_path = args.dwi_path
    bval_path = args.bval_path
    bvec_path = args.bvec_path
    seg_path = args.seg_path
    mask_path = args.mask_path
    annot_path = args.annot_path
    output_path = args.output_path

    if output_path is None:
        dwi_dir = os.path.dirname(dwi_path)
        output_path = os.path.join(dwi_dir, "dsn.csv")

    with open(annot_path) as f:
        labels = json.load(f)
    labels = {int(key): val for key, val in labels.items()}

    arr = compute_dsn_from_file(
        dwi_path,
        bval_path,
        bvec_path, 
        seg_path, 
        list(labels.keys()), 
        mask_path=mask_path
    )

    dsn_df = pd.DataFrame(arr, index=list(labels.values()), columns=list(labels.values()))
    dsn_df.to_csv(output_path)
