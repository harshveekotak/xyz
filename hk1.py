# !pip install opencv-python ultralytics ipython

import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

# Set a preferred camera resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

seen = set()
win_name = "YOLOv8 Live — press q or Esc to quit"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        results = model(frame, conf=0.5, verbose=False)

        # Track unique classes
        r0 = results[0]
        if r0.boxes is not None and r0.boxes.cls is not None:
            for c in r0.boxes.cls:
                seen.add(model.names[int(c)])

        annotated = r0.plot()
        cv2.putText(annotated, "Press q or Esc to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow(win_name, annotated)

        # Quit on 'q' or Esc
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # Quit if window closed
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Summary
if seen:
    print("\nUnique objects detected:")
    for i, name in enumerate(sorted(seen), 1):
        print(f"{i}. {name}")
else:
    print("\nNo objects detected.")
