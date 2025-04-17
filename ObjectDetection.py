from ultralytics import YOLO
import cv2

# Load the YOLOv8 Nano model (super fast and lightweight)
model = YOLO('yolov8n.pt')

# Start webcam capture (default camera is usually 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame, stream=True)

    # Process detection results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f'{model.names[int(class_id)]}: {confidence:.2f}'

            # Draw detection box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the detection results
    cv2.imshow('Real-time YOLOv8 Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()