import cv2
from ultralytics import YOLO
import config

class ObstacleDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def _map_to_grid(self, bbox, frame_width, frame_height):
        """Bounding box középpontjának átszámítása rácskoordinátákká (x, y)."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        grid_x = int((center_x / frame_width) * config.GRID_COLS)
        grid_y = int((center_y / frame_height) * config.GRID_ROWS)

        return {
            "grid_x": min(grid_x, config.GRID_COLS - 1),
            "grid_y": min(grid_y, config.GRID_ROWS - 1),
            "norm_center_x": round(center_x / frame_width, 3),
            "norm_center_y": round(center_y / frame_height, 3)
        }

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # ROI kivágás / maszkolás előkészítése
        roi_frame = frame
        if config.USE_ROI:
            y_min, y_max = int(h * config.ROI_Y_MIN), int(h * config.ROI_Y_MAX)
            x_min, x_max = int(w * config.ROI_X_MIN), int(w * config.ROI_X_MAX)
            roi_frame = frame[y_min:y_max, x_min:x_max]

        results = self.model(
            roi_frame, 
            conf=config.CONFIDENCE_THRESHOLD, 
            verbose=False
        )[0]

        detected_obstacles = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            # Keresett objektumok szűrése (pl: személy, autó, akadály)
            xyxy = box.xyxy[0].cpu().numpy()
            
            if config.USE_ROI:
                xyxy[0] += x_min
                xyxy[2] += x_min
                xyxy[1] += y_min
                xyxy[3] += y_min

            grid_pos = self._map_to_grid(xyxy, w, h)

            detected_obstacles.append({
                "label": class_name,
                "confidence": round(confidence, 2),
                "grid_position": grid_pos,
                "bbox_pixel": [int(v) for v in xyxy]
            })

        return detected_obstacles