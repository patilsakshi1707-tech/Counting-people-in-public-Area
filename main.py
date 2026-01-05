import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def people_counter(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open input video")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    if FPS == 0:
        FPS = 30

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    entered = 0
    exited = 0
    track_history = {}
    counted_ids = set()

    line_y = H // 2
    offset = 25

    print("✅ Processing started... Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame, conf=0.5, classes=[0], verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (0, line_y), (W, line_y), (0, 0, 255), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cy = (t + b) // 2

            prev_cy = track_history.get(track_id, cy)
            track_history[track_id] = cy

            if track_id not in counted_ids:
                if prev_cy < line_y - offset and cy > line_y + offset:
                    entered += 1
                    counted_ids.add(track_id)
                elif prev_cy > line_y + offset and cy < line_y - offset:
                    exited += 1
                    counted_ids.add(track_id)

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        inside = entered - exited

        cv2.putText(frame, f"Entered: {entered}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exited: {exited}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Inside: {inside}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("People Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("✅ Done! Output saved to:", output_path)

# -------------------- RUN --------------------
if __name__ == "__main__":
    print("===================================")
    print(" PEOPLE COUNTER PROGRAM STARTED ")
    print("===================================")

    input_path = input(">>> Enter INPUT video path: ").strip()
    output_path = input(">>> Enter OUTPUT video path: ").strip()

    people_counter(input_path, output_path)
