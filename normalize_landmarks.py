import numpy as np

def normalize_landmarks_step1_shift(landmarks):
    """
    Normalizace krok 1:
    - landmarky jsou list MediaPipe landmarker.landmark
    - nose_index: index bodu nosu (u tebe 4)
    - vrací numpy array tvaru [N, 3] posunutý tak,
      že nos = [0,0,0]
    """
    nose_index = 4  # index nosu v MediaPipe FaceMesh
    
    # převedeme na numpy array [N, 3]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # souřadnice nosu
    nose = pts[nose_index].copy()

    # odečtení
    pts -= nose

    return pts

def normalize_landmarks_step2_scale(pts, right_eye_idx=133, left_eye_idx=362):
    """
    Normalizace KROK 2:
    - pts: numpy array tvaru [N,3] z kroku 1 (nos = [0,0,0])
    - right_eye_idx = 133
    - left_eye_idx  = 362
    Vrací: pts_normalized_scale
    """
    
    # right_eye_idx=133, left_eye_idx=362 (default values from function signature)
    
    # souřadnice koutků očí
    right_eye = pts[right_eye_idx]
    left_eye  = pts[left_eye_idx]

    # vzdálenost mezi nimi (měřítko)
    dist = np.linalg.norm(left_eye - right_eye)

    if dist < 1e-6:
        dist = 1e-6  # ochrana proti dělení nulou

    # škálování
    pts_scaled = pts / dist

    return pts_scaled

import numpy as np

def normalize_landmarks_step3_rotate(pts, right_eye_idx=133, left_eye_idx=362):
    """
    Normalizace KROK 3:
    - vstup: pts tvaru [N, 3] po krocích 1 a 2 (nos v [0,0,0], škálováno)
    - otočí body tak, aby spojnice mezi right_eye_idx a left_eye_idx byla vodorovná
    - rotujeme v rovině (x,y), z necháme beze změny
    """

    # 2D souřadnice koutků očí
    right_eye_xy = pts[right_eye_idx, :2]
    left_eye_xy  = pts[left_eye_idx, :2]

    # vektor z pravého do levého oka
    dx = left_eye_xy[0] - right_eye_xy[0]
    dy = left_eye_xy[1] - right_eye_xy[1]

    # úhel vůči ose x
    theta = np.arctan2(dy, dx)

    # rotační matice pro -theta (aby po otočení byly oči vodorovně)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    # aplikace na všechny (x,y)
    pts_rot = pts.copy()
    xy = pts_rot[:, :2]   # [N,2]
    xy_rot = xy @ R.T     # [N,2]
    pts_rot[:, :2] = xy_rot

    return pts_rot

def normalize_landmarks(landmarks):
    """
    Wrapper function to apply all normalization steps.
    Input: List of [x, y, z] or similar structure.
    Output: Normalized numpy array [N, 3].
    """
    # 1) Shift (Nose to 0,0,0)
    pts = normalize_landmarks_step1_shift(landmarks)
    
    # 2) Scale (Eye distance)
    pts = normalize_landmarks_step2_scale(pts)
    
    # 3) Rotate (Eyes horizontal)
    pts = normalize_landmarks_step3_rotate(pts)
    
    return pts


if __name__ == "__main__":
    # Example usage (only runs when executed directly)
    # 1) posun nosu
    # pts = normalize_landmarks_step1_shift(landmarks, nose_index=4)
    
    # 2) normalizace měřítka podle očí
    # pts = normalize_landmarks_step2_scale(pts, right_eye_idx=133, left_eye_idx=362)
    
    # 2) rotace podle očí
    # pts = normalize_landmarks_step3_rotate(pts, right_eye_idx=133, left_eye_idx=362)
    pass
