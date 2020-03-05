
def detect_faces(video, output, w, h, detect_faces_on_img):
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    vs = cv2.VideoCapture(video)
    i = 0

    while True:
        frame = vs.read()[1]

        if frame is None:
            break

        if i % 3 == 0:
            frame = detect_faces_on_img(frame)
            out.write(frame)
        i += 1

    out.release()

def detect_faces_with_trackers(video, output, w, h, detect_faces_on_img, trackers):
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    vs = cv2.VideoCapture(video)
    i = 0

    while True:
        frame = vs.read()[1]

        if frame is None:
            break

        if i % 10 == 0:
            frame, boxes = detect_faces_on_img(frame)
            for box in boxes:
                trackers.add(cv2.TrackerKCF_create(), frame, box)
        else:
            (success, boxes) = trackers.update(frame)

            # loop over the bounding boxes and draw then on the frame
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)
        i += 1

    out.release()
