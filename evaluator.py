import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config

class DatasetEvaluator:
    def __init__(self, detector):
        self.detector = detector

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def evaluate(self, dataset_path, iou_threshold=0.3):
        path_obj = Path(dataset_path)
        image_paths = sorted(
            list(path_obj.rglob('*.jpg')) + 
            list(path_obj.rglob('*.png')) + 
            list(path_obj.rglob('*.jpeg')) +
            list(path_obj.rglob('*.webp'))
        )

        if not image_paths:
            print("The specified folder does not contain any images.")
            return

        tp = 0
        fn = 0
        fp = 0
        total_gt = 0
        total_pred = 0
        no_detection_count = 0
        low_iou_count = 0

        print(f"\n{len(image_paths)} images being evaluated at the object level (IoU threshold: {iou_threshold})...")

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            if hasattr(config, 'RESIZE_WIDTH') and hasattr(config, 'RESIZE_HEIGHT'):
                resized_frame = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            else:
                resized_frame = frame

            h, w, _ = resized_frame.shape
            
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                alt_label_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
                if alt_label_path.exists():
                    label_path = alt_label_path

            gt_boxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = int((cx - bw / 2) * w)
                            y1 = int((cy - bh / 2) * h)
                            x2 = int((cx + bw / 2) * w)
                            y2 = int((cy + bh / 2) * h)
                            gt_boxes.append([x1, y1, x2, y2])

            total_gt += len(gt_boxes)

            detected_obstacles = self.detector.process_frame(resized_frame)
            pred_boxes = [obj["bbox_pixel"] for obj in detected_obstacles]
            total_pred += len(pred_boxes)

            if len(gt_boxes) > 0 and len(pred_boxes) == 0:
                no_detection_count += len(gt_boxes)

            matched_gt = set()
            matched_pred = set()

            for gt_idx, gt_box in enumerate(gt_boxes):
                best_iou = 0.0
                best_pred_idx = -1
                for pred_idx, pred_box in enumerate(pred_boxes):
                    if pred_idx in matched_pred:
                        continue
                    iou = self._compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(best_pred_idx)
                else:
                    if best_iou > 0:
                        low_iou_count += 1

            fn += (len(gt_boxes) - len(matched_gt))
            fp += (len(pred_boxes) - len(matched_pred))

        precision = tp / (tp + fp + 1e-6)

        report_str = (
            "==================================================\n"
            "           Object-Level Evaluation\n"
            "==================================================\n\n"
            f"Total Ground Truth (GT) boxes:  {total_gt}\n"
            f"Total Predicted boxes:         {total_pred}\n\n"
            f"True Positives (TP):            {tp}\n"
            f"False Negatives (FN):          {fn}\n"
            f"False Positives (FP):             {fp}\n\n"
            f"Precision:                       {precision:.4f}\n"
            "--------------------------------------------------\n"
            "Diagnostic:\n"
            f"- Non-detected obstacles (0 model detections): {no_detection_count}\n"
            f"- Low IoU detections rejected:  {low_iou_count}\n"
            "--------------------------------------------------\n"
        )

        print(report_str)

        with open("evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report_str)

        cm = [[0, fp], [fn, tp]]
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-detected', 'Detected'],
            yticklabels=['Missing', 'Present']
        )
        plt.xlabel('Model detections')
        plt.ylabel('Real obstacles')
        plt.title('Obstacle Detection Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()