import numpy as np
import cv2
from enum import Enum, auto
from imutils.video import VideoStream

import datetime

#from essh_detector import ESSHDetector
from retinaface import RetinaFace
import argparse

# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def init_default():
    emModelPath = 'opencv-face-recognition/face_detection_model'
    protoPath = emModelPath + '/deploy.prototxt'
    # protoPath = './surgery/models/deploy.prototxt'
    modelPath = emModelPath + '/res10_300x300_ssd_iter_140000.caffemodel'


def init_faster_rcnn():
    modelPath = './surgery/models/vgg16_faster_rcnn_iter_80000.caffemodel'
    # protoPath = './surgery/models/deploy_frcnn.prototxt'
    protoPath = './surgery/face-py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/solver.prototxt'


def init_ssh():
    modelPath = './surgery/SSH/data/SSH_models/SSH.caffemodel'
    protoPath = './surgery/SSH/SSH/models/solver_ssh.prototxt'


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


class DSFDModel:
    def __init__(self):
        self.net = cv2.dnn.readNetFromTorch('FaceDetection-DSFD/weights/WIDERFace_DSFD_RES152.pth')


    def forward(self, x):
        return self.net(x)
        


class ESSHModel:
    def __init__(self):
        self.total_time = 0
        self.scales = [1200, 1600]
        self.t = 10
        self.esshDetector = ESSHDetector('./essh/model/essh-r50', 0, ctx_id=0)

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


class DefaultModel:
    def __init__(self):
        emModelPath = 'opencv-face-recognition/face_detection_model'
        protoPath = emModelPath + '/deploy.prototxt'
        # protoPath = './surgery/models/deploy.prototxt'
        modelPath = emModelPath + '/res10_300x300_ssd_iter_140000.caffemodel'
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.number = 4

    def detect_faces(self, video, output):
        detect_faces(video, output, 300, 300, self.detect_faces_on_img)
        return None

    def detect_faces_on_img(self, image):
        image = cv2.resize(image, (300, 300))
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        for detection in detections[0, 0, 0:self.number, 3:7]:
            box = detection * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return image


class RetinaFaceModel:
    def __init__(self):
        gpuid = 0
        self.detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
        self.scales = [1024, 1980]
        self.thresh = 0.8

    def detect_faces(self, video, output):
        detect_faces(video, output, )


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

VIDEO_ROOT = 'videos/'
IMAGE_ROOT = 'images/'


def make_path(img_name):
    return IMAGE_ROOT + img_name


FIRST_VIDEO = 'videos/v1.mp4'
BABY_VIDEO = 'videos/baby2.mp4'
BABY_VIDEO_1 = 'videos/baby1.mp4'
PATIENT_TRANSFER = 'videos/patient_transfer_cut.mp4'


# defaultModel = DefaultModel()
# defaultModel.detect_faces(BABY_VIDEO, 'baby_default.avi')

# esshModel = ESSHModel()
# total_time = esshModel.detect_faces(PATIENT_TRANSFER, 'patient_transfer_essh.avi')
# print('total time: {} seconds'.format(total_time))

# dsfdModel = DSFDModel()
# result = dsfdModel.forward(cv2.imread(IMAGE_ROOT + 'lera.png'))

# print(result)




def parse_args():
    parser = argparse.ArgumentParser(description='Run detectors')
    parser.add_argument('--model', help='Model', default=None, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model == 'retina':
        model = RetinaFaceModel()
    elif args.model == 'essh':
        model = ESSHModel()
    else:
        model = DefaultModel()

    model.detect_faces(BABY_VIDEO, 'baby1_{}.avi'.format(args.model))

if __name__ == '__main__':
    main()
