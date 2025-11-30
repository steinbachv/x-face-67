import numpy as np
import os
from compute_features import compute_polygon_geometry

# Check symetry
def check_symetry(normalized_landmarks, middle_line):
    symetry_indices = load_symetry_points("symetry_points.id")
    ratio = []
    for symetry_row in symetry_indices:
        if 'direction' in symetry_row[0]: 
            half = len(symetry_row[1]) // 2
            left_indices = symetry_row[1][:half]
            right_indices = symetry_row[1][half:]
            angles = []
            for i in range(2):
                if i == 0:
                    symetry_pair = left_indices
                else:
                    symetry_pair = right_indices
                left_idx = symetry_pair[0]
                right_idx = symetry_pair[1]
                left_lm = normalized_landmarks[left_idx]
                right_lm = normalized_landmarks[right_idx]
                a_lm,b_lm = left_lm[0]-right_lm[0], left_lm[1]-right_lm[1]
                # create straight line in XY
                points_x, points_y, points_z = zip(*middle_line)
                a,b = np.polyfit(points_x, points_y, 1)
                angles.append(np.arccos((a_lm +a*b_lm)/(np.sqrt(a_lm**2 + b_lm**2)*np.sqrt(1+a**2))))
            ratio.append(min(angles[0], angles[1]) / max(angles[0], angles[1]))
        if 'polygon' in symetry_row[0]:
            half = len(symetry_row[1]) // 2
            left_indices = symetry_row[1][:half]
            right_indices = symetry_row[1][half:]
            coordinates = normalized_landmarks[:, :2]
            left_pol_coordinates = coordinates[left_indices]
            right_pol_coordinates = coordinates[right_indices]
            left_area = compute_polygon_geometry(left_pol_coordinates)
            right_area = compute_polygon_geometry(right_pol_coordinates)
            ratio.append(min(left_area[2], right_area[2]) / max(left_area[2], right_area[2]))
    return ratio


def load_symetry_points(filename: str = "symetry_points.id") -> list[tuple[str, list[int]]]:
    """
    Načte soubor se symetrickými body.
    Každý řádek ve formátu:
        název,v1,v2,v3,...
    Vrací list tuple:
        [(název, [v1, v2, v3, ...]), ...]
    """
    records: list[tuple[str, list[int]]] = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # přeskoč prázdné řádky
            if not line:
                continue

            parts = line.split(",")
            name = parts[0]
            # zbytek jsou hodnoty – pokusíme se převést na int
            values = [int(p) for p in parts[1:] if p != ""]

            records.append((name, values))

    return records