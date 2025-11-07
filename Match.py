import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

# Convert to homogeneous coordinates
def add_ones(x: np.ndarray):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# Remove outliers
def extractRt(F):
    W = np.array([[0, -1, 0],[1, 0, 0], [0, 0, 1]], dtype=np.float64)
    U, _, Vt = np.linalg.svd(F)

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t

    return ret

def generate_match(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

    # Lowe's ratio test
    ret = []
    x1, x2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1 = f1.key_pts[m.queryIdx]
            pts2 = f2.key_pts[m.trainIdx]

            # Travel is less than 10% of diagonal & is within orb distance 32
            if np.linalg.norm((pts1 - pts2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
                # Keep around indices
                # Reduce O(N²) if possible
                if m.queryIdx not in x1 and m.trainIdx not in x2:
                    x1.append(m.queryIdx)
                    x2.append(m.trainIdx)
                    ret.append((pts1, pts2))

    # Remove duplicates
    assert(len(set(x1)) == len(x1))
    assert(len(set(x2)) == len(x2))

    assert len(ret) >= 8
    ret = np.array(ret)
    x1 = np.array(x1)
    x2 = np.array(x2)

    # RANSAC
    model, f_pts = ransac((ret[:, 0], ret[:, 1]),
                           FundamentalMatrixTransform,
                           min_samples=8,
                           residual_threshold=0.001,
                           max_trials=100)
    print("Matches: %d -> %d -> %d -> %d" % (len(f1.descriptors), len(matches), len(f_pts), sum(f_pts)))

    # Remove outliers
    Rt = extractRt(model.params)

    return x1[f_pts], x2[f_pts], Rt