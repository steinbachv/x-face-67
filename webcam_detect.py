import cv2
import os
import time
import mediapipe as mp
import numpy as np
import datetime
import random
from constants import LABELS, LABELS_idx
import normalize_landmarks
from PIL import Image, ImageDraw, ImageFont
import joblib
from compute_features import compute_features
from load_id import load_indices
from collections import deque



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
DETECTIONS_DIR = "detections"
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to trigger detection
TARGET_WIDTH = 480
TARGET_HEIGHT = 640
# Majority vote buffer (keep last N predictions)
PRED_HISTORY = deque(maxlen=25)


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

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=10):
    """Draw a dashed line between two points."""
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r))
        y = int((pt1[1] * (1 - r) + pt2[1] * r))
        pts.append((x, y))
        
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i+1], color, thickness)

def extract_features(landmarks):
    """
    Extract 16 features from face landmarks using compute_features.py
    
    Args:
        landmarks: MediaPipe face landmarks
        
    Returns:
        numpy array of 16 features (shape: 1, 16)
    """
    # The compute_features function handles normalization internally
    # It expects MediaPipe landmarks directly
    features = compute_features(landmarks, None, None)
    
    # Reshape to (1, 16) for model input
    return features.reshape(1, -1)


def classify_grimace(landmarks, model, label_mapping):
    """
    Classify grimace using SVM model.
    
    Args:
        landmarks: MediaPipe face landmarks
        model: Trained SVM model
        label_mapping: Dict mapping indices to label strings
        
    Returns:
        (predicted_label, confidence, probabilities): Tuple of predicted label string, 
                                                       confidence score (0-1), and class probabilities
    """
    # Extract features using the new function
    features = extract_features(landmarks)
    
    # Get prediction
    prediction = model.predict(features)[0]
    

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities.max()

    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(features)[0]

        # Convert decision function output to a pseudo-probability
        confidence = 1 / (1 + np.exp(-np.max(decision_values)))

        probabilities = None  # unless you compute softmax below
    else:
        confidence = 0.5
        probabilities = None

    
    
    # Map prediction to label string
    labels_predicted = {
        "neutral": "Bez výrazu",
        "smile": "Úsměv",
        "eyes_closed": "Zavřené oči",
        "frown": "Mračíc se",
        "eye_brow": "Zvednuté obočí",
        "mouth_I": "Obličej I",
        "mouth_U": "Obličej U"
    }
    predicted_label = labels_predicted.get(prediction, prediction)
    
    return predicted_label, confidence, probabilities

def calculate_metrics(frame, landmarks):
    """
    Placeholder for metrics calculation.
    
    Args:
        frame: The detected frame
        landmarks: MediaPipe face landmarks
        
    Returns:
        dict: Dictionary of metric names to values
    """
    # TODO: Implement actual metrics calculation
    # For now, return dummy metrics
    return {
        "metric_1": 0.8,
        "metric_2": 0.6,
        "metric_3": 0.9,
        "metric_4": 0.4
    }

def create_classification_info_panel(predicted_label, confidence, probabilities, label_mapping, 
                                     target_width, target_height, font_path):
    """
    Create a visualization panel showing classification information.
    
    Args:
        predicted_label: The predicted grimace label
        confidence: Confidence score (0-1)
        probabilities: Array of probabilities for each class
        label_mapping: Dict mapping indices to label strings
        target_width: Target width for the panel
        target_height: Target height for the panel
        font_path: Path to font file
        
    Returns:
        numpy array: BGR image of the info panel
    """
    # Create black canvas
    panel = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Add title
    panel = put_text_utf8(panel, "Klasifikace", (target_width//2 - 60, 30), 
                         font_path, 28, (255, 255, 255))
    
    # Add predicted label
    instructions_mapping = {
        "neutral": "Bez výrazu",
        "usmev": "Úsměv",
        "zavrit_oci": "Zavřít oči",
        "mrac_se": "Mračení",
        "zvedni_oboci": "Zvednout obočí",
        "I": "I",
        "U": "U"
    }
    
    label_text = instructions_mapping.get(predicted_label, predicted_label)
    panel = put_text_utf8(panel, "Detekce:", (40, 80), 
                         font_path, 24, (255, 255, 255))
    panel = put_text_utf8(panel, f"{label_text}", (40, 120), 
                         font_path, 42, (255, 255, 255))
    
    # Add confidence
    conf_text = f"Jistota: {confidence:.1%}"
    # color = (0, 255, 0) if confidence >= 0.75 else (255, 255, 0) if confidence >= 0.5 else (255, 0, 0)
    panel = put_text_utf8(panel, conf_text, (40, 180), 
                         font_path, 20, (255, 255, 255))
    
    # Add probability bars if available
    if probabilities is not None:
        panel = put_text_utf8(panel, "Pravděpodobnosti:", (30, 170), 
                             font_path, 18, (255, 255, 255))
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        y_offset = 200
        bar_width = target_width - 40
        bar_height = 25
        
        for idx in sorted_indices[:5]:  # Show top 5
            label = label_mapping.get(idx, f"Class {idx}")
            label_display = instructions_mapping.get(label, label)
            prob = probabilities[idx]
            
            # Draw probability bar
            bar_length = int(bar_width * prob)
            cv2.rectangle(panel, (20, y_offset), (20 + bar_width, y_offset + bar_height), 
                         (50, 50, 50), -1)  # Background
            cv2.rectangle(panel, (20, y_offset), (20 + bar_length, y_offset + bar_height), 
                         (0, 255, 0), -1)  # Foreground
            
            # Add text
            text = f"{label_display}: {prob:.1%}"
            panel = put_text_utf8(panel, text, (25, y_offset + 5), 
                                 font_path, 14, (255, 255, 255))
            
            y_offset += bar_height + 10
    
    return panel

def smooth_label(label):
    """
    Majority vote smoothing for labels.
    """
    PRED_HISTORY.append(label)
    # Return the most common label in recent history
    return max(set(PRED_HISTORY), key=PRED_HISTORY.count)

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(DETECTIONS_DIR):
        os.makedirs(DETECTIONS_DIR)
        print(f"Created directory: {DETECTIONS_DIR}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load indices for visualization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pupil_indices = load_indices(os.path.join(script_dir, 'pupils.id'))
    middle_line_indices = load_indices(os.path.join(script_dir, 'middle_line.id'))
    
    print(f"Loaded {len(pupil_indices)} pupil indices: {pupil_indices}")
    print(f"Loaded {len(middle_line_indices)} middle line indices: {middle_line_indices}")

    # Initialize MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Load SVM model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'svm_acc85.joblib')
    
    print(f"Loading SVM model from: {model_path}")
    try:
        svm_model = joblib.load(model_path)
        print("SVM model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create label mapping (index -> label string)
    # Assuming the model was trained with LABELS_idx as the target
    label_mapping = {idx: label for label, idx in zip(LABELS, LABELS_idx)}
    print(f"Label mapping: {label_mapping}")

    print("Starting grimace detection.")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Press 'n' to switch to next grimace.")
    print("Press 'q' to stop.")
    
    # Get script directory for GIF loading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prepare label sequence
    random.seed(time.time())
    label_sequence = list(zip(LABELS, LABELS_idx))
    labels_to_shuffle = label_sequence[1:]
    random.shuffle(labels_to_shuffle)
    label_sequence[1:] = labels_to_shuffle
    
    current_sequence_index = 0
    frame_count = 0
    detection_count = 0
    task_completed = False
    
    # Instructions mapping
    instructions_mapping = {
        "neutral": "Buď bez výrazu",
        "usmev": "Jemně se úsměj",
        "zavrit_oci": "Zavři oči",
        "mrac_se": "Mrač se",
        "zvedni_oboci": "Zvedni obočí",
        "I": "Řekni I",
        "U": "Řekni U"
    }
    
    # GIF Configuration
    grimaces_dir = os.path.join(script_dir, 'grimaces')
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
    
    # Font path
    font_path = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Flip frame horizontally (mirror effect correction)
            frame = cv2.flip(frame, 1)
            
            # Get current target label
            current_label, current_export_id = label_sequence[current_sequence_index]
            
            # --- GIF HANDLING ---
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

            # Resize for consistent processing/display
            processed_frame = resize_with_aspect(frame, TARGET_WIDTH, TARGET_HEIGHT)
            
            # Draw Fixed Cross (Dashed) before processing
            # Position: 0.5 * Width, 0.55 * Height (from bottom)
            center_x = TARGET_WIDTH // 2
            center_y = int(TARGET_HEIGHT * (1-0.55))
            cross_color = (200, 200, 200)  # Light gray
            
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
            
            # Process with MediaPipe
            res = face_mesh.process(rgb)
            
            current_landmarks = None
            predicted_label = None
            confidence = 0.0
            probabilities = None
            # is_detected = False # Removed per-frame flag
            
            if res.multi_face_landmarks:
                for face in res.multi_face_landmarks:
                    current_landmarks = face.landmark
                    h, w = processed_frame.shape[:2]
                    
                    # --- CLASSIFICATION ---
                    raw_label, confidence, probabilities = classify_grimace(current_landmarks, svm_model, label_mapping)

                    # Apply majority-vote smoothing
                    predicted_label = smooth_label(raw_label)                    

                    backward_labels = {
                            "Bez výrazu": "neutral",
                            "Úsměv": "usmev",
                            "Zavřené oči": "zavrit_oci",
                            "Mračíc se": "mrac_se",
                            "Zvednuté obočí": "zvedni_oboci",
                            "Obličej I": "I",
                            "Obličej U": "U"
                        }

                    compare_label = backward_labels.get(predicted_label, predicted_label)
                    # Check if detection matches target and exceeds threshold
                    if compare_label == current_label and confidence >= CONFIDENCE_THRESHOLD:
                        task_completed = True
                        # Calculate metrics for this detection
                        metrics = calculate_metrics(processed_frame, current_landmarks)
                        
                        # Save the detection
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        detection_filename = f"{current_label}_{timestamp}_conf{confidence:.2f}.jpg"
                        detection_path = os.path.join(DETECTIONS_DIR, detection_filename)
                        cv2.imwrite(detection_path, frame)  # Save original frame
                        
                        detection_count += 1
                        print(f"Detection #{detection_count}: {current_label} (confidence: {confidence:.2f}) -> {detection_filename}")
                        

                    
                    # Draw all landmarks on webcam feed (optional - can be commented out)
                    # for lm in face.landmark:
                    #     x, y = int(lm.x * w), int(lm.y * h)
                    #     cv2.circle(processed_frame, (x, y), 1, (0, 255, 0), -1)
                    
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

            # Create classification info panel for third panel
            if predicted_label:
                img_info = create_classification_info_panel(predicted_label, confidence, probabilities, 
                                                            label_mapping, TARGET_WIDTH, TARGET_HEIGHT, font_path)
                # Ensure it matches the target size
                if img_info.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3):
                    img_info = resize_with_aspect(img_info, TARGET_WIDTH, TARGET_HEIGHT)
            else:
                # Black placeholder if no detection
                img_info = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                # Add "Waiting for face..." text
                img_info = put_text_utf8(img_info, "Čekám na detekci obličeje...", (TARGET_WIDTH//6, TARGET_HEIGHT//2), 
                                           font_path, 20, (255, 255, 255))

            frame_count += 1

            # Display current task and status
            instruction_text = instructions_mapping.get(current_label, current_label)
            instruction_color = (0, 255, 0) if task_completed else (255, 255, 255)
            processed_frame = put_text_utf8(processed_frame, f"Úkol: {instruction_text}", (10, 30), 
                                            font_path, 24, instruction_color)
            processed_frame = put_text_utf8(processed_frame, f"Stikni 'n' pro přechod na nový úkol.", (10, 80), 
                                            font_path, 20, (255, 255, 255))
            
            # Create spacers
            spacer_width = 50
            spacer = np.zeros((TARGET_HEIGHT, spacer_width, 3), dtype=np.uint8)

            # Combine [Spacer] [GIF] [Spacer] [Webcam] [Classification Info]
            combined_display = np.hstack((spacer, gif_frame, spacer, processed_frame, img_info))

            # Display the combined frame
            cv2.imshow('Grimace Detection [GIF | Webcam | Classification]', combined_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Move to next grimace
                current_sequence_index = (current_sequence_index + 1) % len(label_sequence)
                task_completed = False
                print(f"Manually switched to: {label_sequence[current_sequence_index][0]}")

    except KeyboardInterrupt:
        print("\nDetection interrupted by user.")

    finally:
        # Clean up
        if current_gif_cap:
            current_gif_cap.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Detection finished. Total detections: {detection_count}. Saved to '{DETECTIONS_DIR}'.")

if __name__ == "__main__":
    main()
