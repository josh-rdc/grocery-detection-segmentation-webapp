
import cv2
import numpy as np

def read_yolo_annotation(file_path):
    """Reads YOLO annotation format with bbox and segmentation and returns structured data."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    segmentations = []
    for line in lines:
        parts = line.strip().split()
        
        # YOLO format: class_id x_center y_center width height [segmentation points...]
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        bbox = (class_id, x_center, y_center, width, height)
        boxes.append(bbox)

        # Check for segmentation points
        if len(parts) > 5:
            segmentation = list(map(float, parts[5:]))
            # Group segmentation points as (x, y) pairs
            segmentation_points = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
            segmentations.append((class_id, segmentation_points))
    
    return boxes, segmentations

def overlay_annotations(image_path, annotations, class_names):
    """Overlay bounding boxes and segmentation masks on the image."""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    boxes, segmentations = annotations

    for seg in segmentations:
    # for box in boxes:
        # class_id, x_center, y_center, w, h = box
        # # Convert YOLO format (center, normalized) to pixel (top-left)
        # x1 = int((x_center - (w / 2)) * width)  # x_min in pixels
        # y1 = int((y_center - (h / 2)) * height)  # y_min in pixels
        # x2 = int((x_center + (w / 2)) * width)  # x_max in pixels
        # y2 = int((y_center + (h / 2)) * height)  # y_max in pixels

        class_id, points = seg
        scaled_points = [(int(x * width), int(y * height)) for x, y in points]
        xs = [p[0] for p in scaled_points]
        ys = [p[1] for p in scaled_points]
        x1, y1 = min(xs), min(ys)  # Top-left corner
        x2, y2 = max(xs), max(ys)  # Bottom-right corner

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (3, 6, 191), 2)

        # label = class_names[class_id]
        # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        label = class_names[class_id]
        # Get the text size to create a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Define the position for the text
        text_x, text_y = x1, y1 - 20

        # Draw a rectangle as background with color #eee5e9 (BGR format: (233, 229, 233))
        cv2.rectangle(image, (text_x, text_y - text_height), (text_x + text_width, text_y + baseline), (233, 229, 233), -1)

        # Draw the black text on top of the background
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    # Draw segmentation masks
    # for seg in segmentations:
    #     class_id, points = seg
        scaled_points = [(int(x * width), int(y * height)) for x, y in points]
        polygon = np.array(scaled_points, np.int32).reshape((-1, 1, 2))

        # Create an overlay for the mask
        overlay = image.copy()

        # Fill the polygon with a semi-transparent color
        mask_color = (114, 116, 254) 
        cv2.fillPoly(overlay, [polygon], mask_color)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # Draw polygon outline
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 255), thickness=2)

        # label = class_names[class_id]
        # centroid = np.mean(polygon.reshape(-1, 2), axis=0).astype(int)
        # cv2.putText(image, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return image