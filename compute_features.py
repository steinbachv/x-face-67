from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from normalize_landmarks import normalize_landmarks

FEATURES_DIR = Path(__file__).with_name("facemask_features")

# Feature list mirrors the new computeFeatures.m order.
FEATURE_SPECS: List[dict] = [
    {"name": "inner_mouth", "file": "inner_mouth.id", "calc_type": "area",
     "fallback": [[13, 312, 311, 310, 415, 306, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 80, 81, 82, 13]]},
    {"name": "left_eye", "file": "left_eye.id", "calc_type": "area",
     "fallback": [[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]]},
    {"name": "right_eye", "file": "right_eye.id", "calc_type": "area",
     "fallback": [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]]},
    {"name": "left_eye_closed", "file": "left_eye_closed.id", "calc_type": "distance",
     "fallback": [[385, 386, 387], [380, 374, 373]]},
    {"name": "right_eye_closed", "file": "right_eye_closed.id", "calc_type": "distance",
     "fallback": [[158, 159, 160], [144, 145, 153]]},
    {"name": "left_eyebrow_up", "file": "left_eyebrow_up.id", "calc_type": "distance",
     "fallback": [[473, 473, 473, 473], [282, 295, 296, 334]]},
    {"name": "right_eyebrow_up", "file": "right_eyebrow_up.id", "calc_type": "distance",
     "fallback": [[468, 468, 468, 468], [52, 65, 66, 105]]},
    {"name": "forehead_height", "file": "forehead_height.id", "calc_type": "distance",
     "fallback": [[9], [151]]},
    {"name": "nose_width", "file": "nose_width.id", "calc_type": "distance",
     "fallback": [[48, 49, 64], [279, 278, 294]]},
    {"name": "mouth_width", "file": "mouth_width.id", "calc_type": "distance",
     "fallback": [[306], [61]]},
    {"name": "nose_to_mouth", "file": "nose_to_mouth.id", "calc_type": "distance",
     "fallback": [[0], [19]]},
    {"name": "mouth_height_to_width", "file": "mouth_width.id", "calc_type": "ratio",
     "fallback": [[13], [14], [61], [291]]},
    {"name": "bottom_lip", "file": "bottom_lip.id", "calc_type": "curvature",
     "fallback": [[306, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]]},
    {"name": "upper_lip", "file": "upper_lip.id", "calc_type": "curvature",
     "fallback": [[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]]},
    {"name": "left_eyebrow_to_eye", "file": "left_eyebrow_to_eye.id", "calc_type": "angle",
     "fallback": [[282, 334], [276, 300], [362, 362], [263, 263]]},
    {"name": "right_eyebrow_to_eye", "file": "right_eyebrow_to _eye.id", "calc_type": "angle",
     "fallback": [[52, 105], [46, 70], [133, 133], [33, 33]]},
]


def compute_polygon_geometry(polydata: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return centroid (Cx, Cy), signed area A and perimeter P for a 2D polygon."""
    pts = np.asarray(polydata, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("polydata must be an Nx2 array.")

    x = pts[:, 0]
    y = pts[:, 1]

    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    step = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = step.sum() / 2.0

    if abs(area) > np.finfo(float).eps:
        cx = ((x[:-1] + x[1:]) * step).sum() / (6 * area)
        cy = ((y[:-1] + y[1:]) * step).sum() / (6 * area)
    else:
        cx = np.mean(x[:-1])
        cy = np.mean(y[:-1])

    perimeter = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).sum()
    return cx, cy, area, perimeter


def _poly_area(xdata: np.ndarray, ydata: np.ndarray, face_area: float | None = None) -> float:
    _, _, area, _ = compute_polygon_geometry(np.column_stack((xdata, ydata)))
    if face_area is None:
        return float(area)
    return float(4 * abs(area) / face_area)


def _median_point(xdata: np.ndarray, ydata: np.ndarray, indices: Iterable[int]) -> Tuple[float, float]:
    idx = np.asarray(list(indices), dtype=int)
    return float(np.median(xdata[idx])), float(np.median(ydata[idx]))


def _med_distance(xdata: np.ndarray, ydata: np.ndarray, groups: List[List[int]]) -> float:
    if len(groups) < 2:
        raise ValueError("Distance feature expects at least two groups of indices.")
    x_a, y_a = _median_point(xdata, ydata, groups[0])
    x_b, y_b = _median_point(xdata, ydata, groups[1])
    return float(np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2))


def _med_ratio(xdata: np.ndarray, ydata: np.ndarray, groups: List[List[int]]) -> float:
    if len(groups) < 4:
        raise ValueError("Ratio feature expects four groups of indices.")
    dist_a = _med_distance(xdata, ydata, groups[0:2])
    dist_b = _med_distance(xdata, ydata, groups[2:4])
    return float(dist_a / dist_b) if dist_b != 0 else np.nan


def _angle_to_direction(xdata: np.ndarray, ydata: np.ndarray, groups: List[List[int]]) -> float:
    if len(groups) < 4:
        raise ValueError("Angle feature expects four groups of indices.")

    x_med = []
    y_med = []
    for group in groups[:4]:
        x_val, y_val = _median_point(xdata, ydata, group)
        x_med.append(x_val)
        y_med.append(y_val)

    v1 = np.array([x_med[1] - x_med[0], y_med[1] - y_med[0]])
    v2 = np.array([x_med[3] - x_med[2], y_med[3] - y_med[2]])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        return np.nan

    dot_prod = float(np.dot(v1, v2))
    angle = np.arccos(np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0))
    return float(angle / np.pi)


def _curvature(xdata: np.ndarray, ydata: np.ndarray) -> float:
    coeffs = np.polyfit(np.asarray(xdata).ravel(), np.asarray(ydata).ravel(), 2)
    return float(coeffs[0])


def _normalize_inputs(xdata_raw: Iterable[float] | Sequence, ydata_raw: Iterable[float] | None, zdata_raw: Iterable[float] | None):
    """
    Normalize landmarks.
    - If only xdata_raw is provided and looks like a list of MediaPipe landmarks, we call normalize_landmarks directly.
    - Otherwise we assume raw x/y(/z) numeric arrays and normalize using nose/eyes.
    """
    # MediaPipe landmarks path
    if ydata_raw is None and hasattr(xdata_raw, "__len__") and len(xdata_raw) > 0 and hasattr(xdata_raw[0], "x"):
        return normalize_landmarks(xdata_raw)

    x_arr = np.asarray(xdata_raw, dtype=float).reshape(-1)
    y_arr = np.asarray(ydata_raw, dtype=float).reshape(-1)

    if x_arr.shape != y_arr.shape:
        raise ValueError("xdata_raw and ydata_raw must have the same shape.")

    if zdata_raw is None:
        z_arr = np.zeros_like(x_arr, dtype=float)
    else:
        z_arr = np.asarray(zdata_raw, dtype=float).reshape(-1)
        if z_arr.shape != x_arr.shape:
            raise ValueError("zdata_raw must match the shape of xdata_raw.")

    pts = np.column_stack((x_arr, y_arr, z_arr))

    nose_idx = 4
    right_eye_idx = 133
    left_eye_idx = 362

    nose = pts[nose_idx].copy()
    pts -= nose

    dist = np.linalg.norm(pts[left_eye_idx] - pts[right_eye_idx])
    dist = dist if dist > 1e-6 else 1e-6
    pts /= dist

    right_eye_xy = pts[right_eye_idx, :2]
    left_eye_xy = pts[left_eye_idx, :2]
    dx = left_eye_xy[0] - right_eye_xy[0]
    dy = left_eye_xy[1] - right_eye_xy[1]
    theta = np.arctan2(dy, dx)

    c, s = np.cos(-theta), np.sin(-theta)
    rotation = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rotation.T

    return pts


def _parse_feature_file(path: Path) -> tuple[str | None, List[List[int]]]:
    """Parse an .id file into an optional type hint and grouped indices."""
    if not path.exists():
        return None, []

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None, []

    type_hint = None
    content_start = 0
    first = lines[0].lower()
    if not any(ch.isdigit() for ch in first):
        type_hint = first
        content_start = 1

    raw = ";".join(lines[content_start:])
    groups: List[List[int]] = []
    for group_str in raw.split(";"):
        tokens = group_str.replace(",", " ").split()
        group = [int(tok) for tok in tokens if tok.lstrip("+-").isdigit()]
        if group:
            groups.append(group)

    return type_hint, groups


def load_facemask_feature_data(features_dir: Path | str | None = None) -> dict:
    """Load all .id feature files into a dict for reuse (visualization + computation)."""
    base_dir = Path(features_dir) if features_dir else FEATURES_DIR
    data: dict = {}
    if not base_dir.exists():
        return data

    for file in base_dir.glob("*.id"):
        type_hint, groups = _parse_feature_file(file)
        flat = [idx for grp in groups for idx in grp]
        data[file.stem] = {"type_hint": type_hint, "groups": groups, "flat_indices": flat}
    return data


def _get_feature_groups(name: str, file_name: str, fallback: List[List[int]], calc_type: str) -> List[List[int]]:
    path = FEATURES_DIR / file_name
    _, groups = _parse_feature_file(path)
    expected_counts = {"distance": 2, "ratio": 4, "angle": 4}
    expected_len = expected_counts.get(calc_type)

    if (not groups) or (expected_len and len(groups) != expected_len):
        return fallback
    return groups


def compute_features(xdata_raw: Iterable[float], ydata_raw: Iterable[float], zdata_raw: Iterable[float] | None = None) -> np.ndarray:
    """
    Python port of the updated computeFeatures.m.

    Inputs are raw landmark coordinates (iterables of x, y, z) in MediaPipe indexing (0-based).
    Returns a numpy array of 16 features following the MATLAB ordering.
    """
    normalized = _normalize_inputs(xdata_raw, ydata_raw, zdata_raw)
    xdata = normalized[:, 0]
    ydata = normalized[:, 1]

    # Face contour for area normalization
    face_groups = _get_feature_groups(
        "face",
        "face.id",
        [[10, 338, 297, 332, 284, 251, 389, 356, 447, 454, 366, 323, 401, 361, 435, 288,
          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 215, 132,
          93, 234, 127, 162, 21, 54, 103, 67, 109, 10]],
        "area",
    )
    face_area = _poly_area(xdata[face_groups[0]], ydata[face_groups[0]])

    features = np.zeros(len(FEATURE_SPECS), dtype=float)

    for idx, spec in enumerate(FEATURE_SPECS):
        groups = _get_feature_groups(spec["name"], spec["file"], spec["fallback"], spec["calc_type"])
        calc_type = spec["calc_type"]

        if calc_type == "area":
            features[idx] = _poly_area(xdata[groups[0]], ydata[groups[0]], face_area)
        elif calc_type == "distance":
            features[idx] = _med_distance(xdata, ydata, groups)
        elif calc_type == "ratio":
            features[idx] = _med_ratio(xdata, ydata, groups)
        elif calc_type == "angle":
            features[idx] = _angle_to_direction(xdata, ydata, groups)
        elif calc_type == "curvature":
            features[idx] = _curvature(xdata[groups[0]], ydata[groups[0]])
        else:
            raise ValueError(f"Unknown calculation type {calc_type} for feature {spec['name']}")

    return features


__all__ = ["compute_features", "compute_polygon_geometry", "load_facemask_feature_data"]
