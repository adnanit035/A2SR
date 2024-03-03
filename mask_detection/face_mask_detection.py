from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


class MaskDetector:
    def __init__(self, face_mask_detector, mask_detector_model):
        self._base = MaskDetection_MobileNet(face_mask_detector, mask_detector_model)

    def detect_face_mask(self, frame, cv2):
        try:
            return self._base.detect_face_mask(frame, cv2)
        except:
            return "Unknown", 0


class MaskDetection_MobileNet:

    def __init__(self, face_mask_detector, mask_detector_model):
        self.face_detector = face_mask_detector
        self.mask_detector_model = mask_detector_model

    def detect_face_mask(self, frame, cv2):
        # grab the dimensions of the frame and then construct a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel,
                # resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.mask_detector_model.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding locations
        return (locs, preds)
