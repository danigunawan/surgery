import numpy as np
import cv2
from retinaface import RetinaFace
import argparse
import os
import common


class RetinaFaceModel:
    def __init__(self, with_tracking=False, thresh=0.8, fpd=10):
        gpuid = 0
        self.detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        self.scales = [1024, 1980]
        self.thresh = thresh
        self.with_tracking = with_tracking
        self.frames_per_detection = fpd
        print('initialized retina face model')

    def detect_faces(self, video, output):
        if self.with_tracking:
            common.detect_faces_with_trackers(video, output, 640, 360, self.detect_faces_on_img)
        else:
            common.detect_faces(video, output, 640, 360, self.detect_faces_on_img)

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
            for face in faces:
                image = common.blur(image, face.astype(np.int))
        return image, [(face[0], face[1], face[2] - face[0], face[3] - face[1]) for face in faces]


def parse_args():
    parser = argparse.ArgumentParser(description='Run detectors')
    parser.add_argument('--video', help='Video', default=None, type=str)
    parser.add_argument('--vroot', help='Video root', default=None, type=str)
    parser.add_argument('--tracking', help='with tracking', default=None, type=str)
    parser.add_argument('--threshold', help='threshold', default=0.8, type=float)
    parser.add_argument('-s', help='Save video with detections directory', default=None, type=str)
    parser.add_argument('--save_path', help='Save video to the spec path', default=None, type=str)
    parser.add_argument('--fpd', help='Frames to be tracked per each detection', default=None, type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with_tracking = args.tracking is not None and args.tracking == 'y'
    fpd = args.fpd
    if fpd is None:
        fpd = 10
    model = RetinaFaceModel(with_tracking, args.threshold, fpd=fpd)

    if args.save_path is None:
        output_name_constructor = lambda src_video_name: args.s + os.sep + '{}_retina_detected.avi'.format(
            src_video_name[:-4])
    else:
        output_name_constructor = lambda src_video_name: '{}_retina_detected.avi'.format(args.save_path)

    if args.video is not None:
        model.detect_faces(args.video, output_name_constructor(args.video.split('/')[-1]))
    else:
        videos = os.listdir(args.vroot)

        for video in videos:
            print('======\nPocessing video : {}\n======'.format(video))
            model.detect_faces(args.vroot + os.sep + video,
                               output_name_constructor(video))


if __name__ == '__main__':
    main()