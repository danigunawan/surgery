import numpy as np
import cv2

import datetime

from essh_detector import ESSHDetector
import argparse



def detect_faces(video, output, w, h, detect_faces_on_img):
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    result_video = []
    vs = cv2.VideoCapture(video)
    i = 0

    while True and i < 1000:
        frame = vs.read()[1]

        if frame is None:
            break

        if i % 3 == 0:
            frame = detect_faces_on_img(frame)
            result_video.append(frame)
            out.write(frame)
        i += 1

    for frame in result_video:
        cv2.imshow('surgery', frame)
        key = cv2.waitKey(100)

        if key == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()



class ESSHModel:
    def __init__(self):
        self.total_time = 0
        self.scales = [1200, 1600]
        self.t = 10
        self.esshDetector = ESSHDetector('./model/essh-r50', 0, ctx_id=-1)

    def detect_faces(self, video, output):
        self.total_time = 0
        detect_faces(video, output, 640, 360, self.detect_faces_on_img)
        return self.total_time

    def detect_faces_on_img(self, img):
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if im_size_min > target_size or im_size_max > max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
            print('resize to', img.shape)

        # for i in range(t - 1):  # warmup
        #     faces = esshDetector.detect(img)
        timea = datetime.datetime.now()

        faces = self.esshDetector.detect(img, threshold=0.5)
        # print("DETECTED!!!!!!!")
        bbox = np.round(faces[:, 0:5])
        landmark = faces[:, 5:15].reshape(-1, 5, 2)

        timeb = datetime.datetime.now()
        diff = timeb - timea
        diff = diff.total_seconds()
        self.total_time += diff
        # print('detection uses', diff.total_seconds(), 'seconds')
        # print('find', faces.shape[0], 'faces')

        for b in bbox:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        for p in landmark:
            for i in range(5):
                cv2.circle(img, (p[i][0], p[i][1]), 1, (0, 0, 255), 2)
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='Run detectors')
    parser.add_argument('--video', help='Video', default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = ESSHModel()

    model.detect_faces(args.video, 'baby1_{}.avi'.format('essh'))


if __name__ == '__main__':
    main()
