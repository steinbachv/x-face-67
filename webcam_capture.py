import cv2
import os
import time
import mediapipe as mp
import numpy as np
import csv
import datetime
import random
from constants import LABELS, LABELS_idx
import normalize_landmarks
from PIL import Image, ImageDraw, ImageFont

def put_text_utf8(img, text, position, font_path, font_size, color):
    """
    Draws text with UTF-8 characters using PIL.
    img: OpenCV image (BGR)
    text: String to draw
    position: (x, y) tuple
    font_path: Path to .ttf file
    font_size: Integer
    color: (R, G, B) tuple
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default font if path is invalid
        print(f"Warning: Font not found at {font_path}. Using default.")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "frame"
FACES_DIR = "faces"
DATASET_DIR = "dataset"
FPS = 30       # Desired framerate (approximate)
TARGET_WIDTH = 480
TARGET_HEIGHT = 640

def resize_with_aspect(image, target_width, target_height):
    h, w = image.shape[:2]

    # jednotný scale faktor pro zachování poměru stran
    scale = min(target_width / w, target_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # poměrový resize
    resized = cv2.resize(image, (new_w, new_h))

    # vytvořit plátno (černé pozadí)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # zarovnání na střed
    x_start = (target_width - new_w) // 2
    y_start = (target_height - new_h) // 2

    # vložit obrázek doprostřed
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    
    return canvas

def load_feature(filename):
    """
    Loads feature indices from a file.
    Returns: (type, indices)
    type: 'polygon' or 'line'
    indices: list of integers
    """
    indices = []
    feature_type = 'polygon' # Default
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            if not lines:
                return feature_type, indices
                
            first_line = lines[0].strip().lower()
            content_start_idx = 0
            
            if first_line in ['line', 'polygon']:
                feature_type = first_line
                content_start_idx = 1
            
            # Process remaining lines
            content = "".join(lines[content_start_idx:])
            # Replace common separators with comma
            content = content.replace('\n', ',').replace('.', ',')
            parts = content.split(',')
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    indices.append(int(p))
    else:
        print(f"Warning: File '{filename}' not found.")
        
    return feature_type, indices

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=10):
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r))
        y = int((pt1[1] * (1 - r) + pt2[1] * r))
        pts.append((x, y))
        
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i+1], color, thickness)

def main():
    # Create output directories if they don't exist
    # if not os.path.exists(OUTPUT_DIR):
        # os.makedirs(OUTPUT_DIR)
        # print(f"Created directory: {OUTPUT_DIR}")
        
    # if not os.path.exists(FACES_DIR):
        # os.makedirs(FACES_DIR)
        # print(f"Created directory: {FACES_DIR}")
        
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created directory: {DATASET_DIR}")

    # Load indices for visualization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _, pupil_indices = load_feature(os.path.join(script_dir, 'pupils.id'))
    _, middle_line_indices = load_feature(os.path.join(script_dir, 'middle_line.id'))
    
    # Load Facemask Features
    features_dir = os.path.join(script_dir, 'facemask_features')
    feature_data = {} # name -> {'type': type, 'indices': indices}
    
    if os.path.exists(features_dir):
        for f in os.listdir(features_dir):
            if f.endswith('.id'):
                name = os.path.splitext(f)[0]
                ftype, findices = load_feature(os.path.join(features_dir, f))
                feature_data[name] = {'type': ftype, 'indices': findices}
                print(f"Loaded feature '{name}': Type={ftype}, Points={len(findices)}")
    
    print(f"Loaded {len(pupil_indices)} pupil indices: {pupil_indices}")
    print(f"Loaded {len(middle_line_indices)} middle line indices: {middle_line_indices}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize Matplotlib Figure
    # Calculate DPI to match TARGET_WIDTH/HEIGHT roughly
    dpi = 100
    fig_width = TARGET_WIDTH / dpi
    fig_height = TARGET_HEIGHT / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # White axis lines, ticks, and labels
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Define colors for features
    features = [
        'face', 'left_eye', 'right_eye', 'inner_mouth',
        'left_eye_brow', 'right_eye_brow', 'nose_width',
        'symetry_right', 'symetry_left', 'mouth_height_to_width',
        'jawline', 'bottom_lip', 'frown_right', 'frown_left'
    ]
    features = [
            'bottom_lip', 'face_right', 'frown_right', 'left_eye_brow', 'left_eye_closed', 'mouth_width',
            'right_eye_brow', 'right_eye_closed', 'symetry_right', 'face', 'forehead_height', 'inner_mouth',
            'left_eyebrow_to_eye', 'left_eye', 'nose_to_mouth', 'right_eyebrow_to _eye', 'right_eye',
            'face_left', 'upper_lip', 'frown_left', 'jawline', 'left_eyebrow_up',' mouth_height_to_width',
            'nose_width', 'right_eyebrow_up', 'symetry_left'
            ]

    # Pick a colormap
    cmap = plt.get_cmap("hsv")

    # Generate evenly spaced colors between 0–1
    values = np.linspace(0, 1, len(features))

    feature_colors = {feat: cmap(val) for feat, val in zip(features, values)}
    
    # Create Patches (Polygons and Lines)
    patches = {}
    from matplotlib.patches import Polygon
    
    # Draw face first (background) if it exists
    if 'face' in feature_data:
        data = feature_data['face']
        ftype = data['type']
        color = feature_colors.get('face', 'gray')
        
        if ftype == 'polygon':
            poly = Polygon(np.zeros((1, 2)), closed=True, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(poly)
            patches['face'] = poly
        elif ftype == 'line':
             line, = ax.plot([], [], color=color, linewidth=2)
             patches['face'] = line
        
    for name, data in feature_data.items():
        if name == 'face': continue # Already added
        
        color = feature_colors.get(name, (0, 1, 0, 0.5)) # Default Green
        ftype = data['type']
        
        if ftype == 'polygon':
            poly = Polygon(np.zeros((1, 2)), closed=True, facecolor=color, edgecolor=color[:3], linewidth=1)
            ax.add_patch(poly)
            patches[name] = poly
        elif ftype == 'line':
            # For lines, we use ax.plot which returns a list of Line2D objects
            line, = ax.plot([], [], color=color, linewidth=2)
            patches[name] = line

    scat = ax.scatter([], [], s=2, c='white', alpha=0.5) # Smaller, white points
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(2.5, -2.5) # Invert Y to match image coordinates
    ax.set_title("Normalized Face", color='white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Pre-allocate buffer for plot image
    fig.canvas.draw() # Initial draw

    print("Starting continuous capture.")
    print("Press 'c' to capture dataset entry (Frame + CSV).")
    print("Press 'q' to stop.")
    
    # Prepare randomized label sequence
    random.seed(time.time())
    label_sequence = list(zip(LABELS, LABELS_idx))
    labels_to_shuffle = label_sequence[1:]
    random.shuffle(labels_to_shuffle)
    label_sequence[1:] = labels_to_shuffle
    
    current_sequence_index = 0
    frame_count = 0
    # Track capture count per label
    label_capture_count = {label: 0 for label in LABELS}

    try:
        # GIF Configuration
        grimaces_dir = os.path.join(script_dir, 'grimaces')
        # Map labels to filenames (adjust as needed)
        gif_mapping = {
            "neutral": "neutral.gif",
            "usmev": "usmev.gif",
            "zavrit_oci": "zavri_oci.gif",
            "mrac_se": "mrac_se.gif",
            "zvedni_oboci": "zvedni_oboci.gif",
            "I": "I.gif",
            "U": "U.gif"
        }
        
        current_gif_cap = None
        current_gif_label = None

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Flip frame horizontally (mirror effect correction)
            frame = cv2.flip(frame, 1)
            
            # --- GIF HANDLING ---
            current_label, current_export_id = label_sequence[current_sequence_index]
            
            # Check if we need to load a new GIF
            if current_label != current_gif_label:
                if current_gif_cap:
                    current_gif_cap.release()
                
                gif_filename = gif_mapping.get(current_label)
                if gif_filename:
                    gif_path = os.path.join(grimaces_dir, gif_filename)
                    if os.path.exists(gif_path):
                        current_gif_cap = cv2.VideoCapture(gif_path)
                        current_gif_label = current_label
                    else:
                        print(f"Warning: GIF not found: {gif_path}")
                        current_gif_cap = None
                        current_gif_label = None
                else:
                    current_gif_cap = None
                    current_gif_label = None
            
            # Read GIF frame
            gif_frame = None
            if current_gif_cap and current_gif_cap.isOpened():
                ret_gif, gif_frame_raw = current_gif_cap.read()
                if not ret_gif:
                    # Loop back to start
                    current_gif_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_gif, gif_frame_raw = current_gif_cap.read()
                
                if ret_gif:
                    # Resize GIF to match TARGET_WIDTH and TARGET_HEIGHT exactly (same as webcam)
                    gif_frame = resize_with_aspect(gif_frame_raw, TARGET_WIDTH, TARGET_HEIGHT)
            
            # If no GIF available, create black placeholder
            if gif_frame is None:
                gif_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

            # --- REAL-TIME ANALYSIS ---
            # Resize for consistent processing/display
            processed_frame = resize_with_aspect(frame, TARGET_WIDTH, TARGET_HEIGHT)
            
            # Draw Fixed Cross (Dashed)
            # Position: 0.5 * Width, 0.65 * Height
            center_x = TARGET_WIDTH // 2
            center_y = int(TARGET_HEIGHT * (1-0.55))
            cross_color = (200, 200, 200) # Light gray/White
            
            # Calculate video bounds within the canvas to avoid drawing on black bars
            h_orig, w_orig = frame.shape[:2]
            scale = min(TARGET_WIDTH / w_orig, TARGET_HEIGHT / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            x_start = (TARGET_WIDTH - new_w) // 2
            y_start = (TARGET_HEIGHT - new_h) // 2
            
            # Horizontal line (constrained to video width)
            draw_dashed_line(processed_frame, (x_start, center_y), (x_start + new_w, center_y), cross_color, 1, 15)
            # Vertical line (constrained to video height)
            draw_dashed_line(processed_frame, (center_x, y_start), (center_x, y_start + new_h), cross_color, 1, 15)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Process
            res = face_mesh.process(rgb)
            
            current_landmarks = None
            
            if res.multi_face_landmarks:
                for face in res.multi_face_landmarks:
                    current_landmarks = face.landmark
                    h, w = processed_frame.shape[:2]
                    
                    # --- NORMALIZATION ---
                    # Normalize directly from MediaPipe landmarks
                    normalized_landmarks = normalize_landmarks.normalize_landmarks(face.landmark)
                    
                    # Update Matplotlib Figure
                    # normalized_landmarks is [N, 3]
                    # ax.set_xlim and ax.set_ylim are set once in initialization
                    scat.set_offsets(normalized_landmarks[:, :2])
                    
                    # Update Patches (Polygons and Lines)
                    for name, patch in patches.items():
                        data = feature_data.get(name)
                        if data:
                            indices = data['indices']
                            ftype = data['type']
                            
                            if indices:
                                # Filter indices to ensure they are within bounds
                                valid_indices = [i for i in indices if i < len(normalized_landmarks)]
                                if valid_indices:
                                    points = normalized_landmarks[valid_indices, :2]
                                    
                                    # Check the type of the patch object
                                    if isinstance(patch, Polygon):
                                        patch.set_xy(points)
                                    else:
                                        # Assume Line2D
                                        patch.set_data(points[:, 0], points[:, 1])
                    
                    # Render plot to buffer
                    fig.canvas.draw()
                    
                    # Convert to numpy array (RGBA)
                    img_plot = np.array(fig.canvas.buffer_rgba())
                    
                    # Convert RGBA to BGR for OpenCV
                    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                    
                    # Resize to match processed_frame height if needed (though we tried to match it via figsize)
                    if img_plot.shape[0] != processed_frame.shape[0]:
                         img_plot = resize_with_aspect(img_plot, TARGET_WIDTH, TARGET_HEIGHT)
                    
                    # --- END NORMALIZATION ---

                    # Draw all landmarks on Webcam Feed
                    for lm in face.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(processed_frame, (x, y), 1, (0, 255, 0), -1)
                        
                    # Draw Pupil Line (Blue)
                    if len(pupil_indices) >= 2:
                        p1_idx = pupil_indices[0]
                        p2_idx = pupil_indices[1]
                        if p1_idx < len(face.landmark) and p2_idx < len(face.landmark):
                            p1 = face.landmark[p1_idx]
                            p2 = face.landmark[p2_idx]
                            x1, y1 = int(p1.x * w), int(p1.y * h)
                            x2, y2 = int(p2.x * w), int(p2.y * h)
                            cv2.line(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                    # Draw Middle Line (Red)
                    if len(middle_line_indices) > 1:
                        points = []
                        for idx in middle_line_indices:
                            if idx < len(face.landmark):
                                lm = face.landmark[idx]
                                points.append((int(lm.x * w), int(lm.y * h)))
                        
                        if len(points) > 1:
                            # Draw lines connecting sequence of points
                            for i in range(len(points) - 1):
                                cv2.line(processed_frame, points[i], points[i+1], (0, 0, 255), 2)
            else:
                # If no face, create a blank placeholder for the plot
                img_plot = np.zeros_like(processed_frame)

            # Save processed frame continuously to faces/
            # faces_frame_name = os.path.join(FACES_DIR, f"frame_{frame_count}.jpg")
            # cv2.imwrite(faces_frame_name, processed_frame)
            frame_count += 1

            # Display instructions
            # current_label already fetched above
            instructions_mapping = {
                "neutral": "Buď bez výrazu",
                "usmev": "Jemně se úsměj",
                "zavrit_oci": "Zavři oči",
                "mrac_se": "Mrač se",
                "zvedni_oboci": "Zvedni obočí",
                "I": "Řekni I",
                "U": "Řekni U"
            }
            
            font_path = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"
            processed_frame = put_text_utf8(processed_frame, f"{instructions_mapping[current_label]}", (10, 50), 
                                            font_path, 32, (255, 255, 255))
            processed_frame = put_text_utf8(processed_frame, "Zmáčkni 'c' pro snímnutí.", (10, 90), 
                                            font_path, 22, (255, 255, 255))
            
            # Combine GIF, Webcam, and Plot View
            # Ensure img_plot is defined even if no face (handled above)
            if 'img_plot' not in locals():
                 img_plot = np.zeros_like(processed_frame)
                 
            # Resize img_plot to match processed_frame dimensions exactly for hstack
            if img_plot.shape != processed_frame.shape:
                img_plot = cv2.resize(img_plot, (processed_frame.shape[1], processed_frame.shape[0]))
            

            # Create a black spacer
            spacer_width = 50
            spacer = np.zeros((processed_frame.shape[0], spacer_width, 3), dtype=np.uint8)

            # Add spacer to the left of GIF as well
            combined_display = np.hstack((spacer, gif_frame, spacer, processed_frame, img_plot))

            # Display the combined frame
            cv2.imshow('Webcam Capture - FaceMesh (GIF | Raw | Normalized)', combined_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture dataset entry
                if current_landmarks:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    base_filename = f"{current_label}_{timestamp}"
                    
                    # Create clean version (without circles)
                    clean_frame = resize_with_aspect(frame, TARGET_WIDTH, TARGET_HEIGHT)
                    
                    # Save MASKED version to dataset/
                    masked_filename = f"{base_filename}_masked.jpg"
                    masked_path = os.path.join(DATASET_DIR, masked_filename)
                    cv2.imwrite(masked_path, combined_display)
                    
                    # Save CLEAN version to dataset/
                    clean_filename = f"{base_filename}_clean.jpg"
                    clean_path = os.path.join(DATASET_DIR, clean_filename)
                    cv2.imwrite(clean_path, frame) # Original flipped frame
                    
                    # Append to label-specific CSV
                    # Format like pokus2.py: [pom, x, y, z, pom, x, y, z, ...]
                    label_csv_path = os.path.join(DATASET_DIR, f"{current_label}.csv")
                    label_capture_count[current_label] += 1
                    pom = current_export_id # Use the consistent ID from constants.py
                    
                    # Flatten landmarks: [x1, y1, z1, x2, y2, z2, ...]
                    landmarks_flat = []
                    for lm in current_landmarks:
                        landmarks_flat.extend([lm.x, lm.y, lm.z])
                        
                    # Create row: [pom, landmarks...]
                    row = [pom] + landmarks_flat
                    
                    # Write to CSV
                    with open(label_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    print(f"Captured {current_label} (ID: {pom}) -> {base_filename}")
                    
                    # Move to next label
                    current_sequence_index = (current_sequence_index + 1) % len(label_sequence)
                else:
                    print("No face detected! Cannot capture.")

    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")

    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        # print(f"Capture finished. Saved {frame_count} frames to '{FACES_DIR}'. Dataset saved to '{DATASET_DIR}'.")
        print(f"Capture finished. Dataset saved to '{DATASET_DIR}'.")

if __name__ == "__main__":
    main()
