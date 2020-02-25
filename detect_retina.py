import numpy as np
import cv2
from retinaface import RetinaFace
import argparse
import os

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

class RetinaFaceModel:
    def __init__(self):
        gpuid = 0
        self.detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        self.scales = [1024, 1980]
        self.thresh = 0.8
        print('initialized retina face model')

    def detect_faces(self, video, output):
        detect_faces(video, output, 640, 360, self.detect_faces_on_img)

    def detect_faces_on_img(self, image):
        im_shape = image.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        # im_scale = 1.0
        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]
        flip = False

        faces, landmarks = self.detector.detect(image, self.thresh, scales=scales, do_flip=flip)
        if faces is not None:
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

        print(image.shape)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description='Run detectors')
    parser.add_argument('--video', help='Video', default=None, type=str)
    parser.add_argument('--vroot', help='Video root', default=None, type=str)
    parser.add_argument('-s', help='Save video with detections directory', default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = RetinaFaceModel()

    if args.video is not None:
        model.detect_faces(args.video, '{}_retina_detected.avi'.format(args.video))
    else:
        videos = os.listdir(args.vroot)

        for video in videos:
            print('======\nPocessing video : {}\n======'.format(video))
            model.detect_faces(args.vroot + os.sep + video,
                               args.s + os.sep + '{}_retina_detected.avi'.format(video[:-4]))



if __name__ == '__main__':
    main()
