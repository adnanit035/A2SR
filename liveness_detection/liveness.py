import numpy as np
from enum import Enum
import cv2
from imutils import face_utils
import dlib
from scipy.spatial.distance import euclidean

class FaceLivenessModels(Enum):
    EYESBLINK = 0  # depends on multiple frames
    COLORSPACE_YCRCBLUV = 1  # depends on single frame only
    DEFAULT = EYESBLINK


class FaceLiveness:
    def __init__(self, model=FaceLivenessModels.DEFAULT, path=None):
        if model == FaceLivenessModels.EYESBLINK:
            self.base1 = FaceLiveness_EYESBLINK(path)
        elif model == FaceLivenessModels.COLORSPACE_YCRCBLUV:
            self.base2 = FaceLiveness_COLORSPACE_YCRCBLUV(path)

    def is_eyes_close(self, frame, face):
        return self.base1.is_eyes_close(frame, face)

    def set_eye_threshold(self, threshold):
        self.base1.set_eye_threshold(threshold)

    def get_eye_threshold(self):
        return self.base1.get_eye_threshold()

    def is_eyes_close_true(self, frame, face):
        eyes_close, eyes_ratio = self.base1.is_eyes_close(frame, face)
        
        return eyes_close, eyes_ratio

    def is_fake_print_fake_replay(self, frame, face):
        print_attack = self.base2.is_print_attack(frame, face)
        reply_attack = self.base2.is_reply_attack(frame, face)

        return print_attack, reply_attack


class FaceLiveness_EYESBLINK:
    ear_threshold = 0.3  # eye aspect ratio (ear); less than this value, means eyes is close
    ear_consecutive_frames = 3

    def __init__(self, path):
        self.detector = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')
        (self.leye_start, self.leye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.reye_start, self.reye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        

    def is_eyes_close(self, frame, face):
        shape = self.get_shape(frame, face)
        average_ear = (self.eye_aspect_ratio(shape[self.leye_start:self.leye_end])
                       + self.eye_aspect_ratio(shape[self.reye_start:self.reye_end])
                      ) / 2.0
        return (average_ear < self.ear_threshold), average_ear

    def set_eye_threshold(self, threshold):
        self.ear_threshold = threshold

    def get_eye_threshold(self):
        return self.ear_threshold

    def eye_aspect_ratio(self, eye):
        # (|p1-p5|+|p2-p4|) / (2|p0-p3|)
        # np.linalg.norm is faster than dist.euclidean
        # return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (
        #         2.0 * np.linalg.norm(eye[0] - eye[3]))
        return (euclidean(eye[1],eye[5]) + euclidean(eye[2],eye[4])) / (2.0 * euclidean(eye[0],eye[3]))

    def get_shape(self, frame, face):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = self.detector(frame_gray, rect)
        coords = np.zeros((shape.num_parts, 2), dtype="int")
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords


class FaceLiveness_COLORSPACE_YCRCBLUV:
    threshold_print = 0.35
    threshold_replay = 0.93

    def __init__(self, path):
        from sklearn.externals import joblib
        try:
            self.clf_print = joblib.load(path + "colorspace_ycrcbluv_print.pkl")
        except Exception as e:
            print("FaceLiveness_COLORSPACE_YCRCBLUV joblib exception {}".format(e))
        try:
            self.clf_replay = joblib.load(path + "colorspace_ycrcbluv_replay.pkl")
        except Exception as e:
            print("FaceLiveness_COLORSPACE_YCRCBLUV joblib2 exception {}".format(e))

    def is_print_attack(self, frame, face):
        feature_vector = self.get_embeddings(frame, face)
        prediction = self.clf_replay.predict_proba(feature_vector)
        if np.mean(prediction[0][1]) >= self.threshold_print:
            return True
        return False

    def is_reply_attack(self, frame, face):
        feature_vector = self.get_embeddings(frame, face)
        prediction = self.clf_replay.predict_proba(feature_vector)
        if np.mean(prediction[0][1]) >= self.threshold_replay:
            return True
        return False

    def get_embeddings(self, frame, face):
        (x, y, w, h) = face
        img = frame[y:y + h, x:x + w]
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        hist_ycrcb = self.calc_hist(img_ycrcb)
        hist_luv = self.calc_hist(img_luv)
        feature_vector = np.append(hist_ycrcb.ravel(), hist_luv.ravel())
        return feature_vector.reshape(1, len(feature_vector))

    def calc_hist(self, img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
