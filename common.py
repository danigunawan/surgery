import cv2
import numpy as np


def detect_faces(video, output, w, h, detect_faces_on_img):
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    vs = cv2.VideoCapture(video)
    i = 0

    while True:
        frame = vs.read()[1]

        if frame is None:
            break

        if True:
            frame, _ = detect_faces_on_img(frame)
            out.write(frame)
        i += 1

    out.release()
    print("video saved to " + output)


def detect_faces_with_trackers(video, output, w, h, detect_faces_on_img):
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    vs = cv2.VideoCapture(video)
    i = 0


    trackers = None
    while True:
        frame = vs.read()[1]

        if frame is None:
            break

        if i % 10 == 3:
            trackers = cv2.MultiTracker_create()
            frame, boxes = detect_faces_on_img(frame)
            for box in boxes:
                trackers.add(cv2.TrackerKCF_create(), frame, box)
        elif trackers is not None:
            (success, boxes) = trackers.update(frame)

            # loop over the bounding boxes and draw then on the frame
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)
        i += 1

    out.release()


def blur(frame, box):
    print('started blurring')
    h, w, _ = frame.shape

    blurred = cv2.blur(frame, (50, 50))

    mask = np.zeros(frame.shape, dtype=np.uint8)
    mask = cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

    out = np.where(mask == (255, 255, 255), blurred, frame)
    print('finished blurring')
    return out


def draw_detections(image, faces, landmarks):
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        # print('score', faces[i][4])
        box = faces[i].astype(np.int)
        # color = (255,0,0)
        color = (0, 0, 255)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            # print(landmark.shape)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(image, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
