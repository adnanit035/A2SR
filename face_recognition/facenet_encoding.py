import os
import numpy as np
import cv2
import facenet
from scipy import misc
from sklearn.externals import joblib
from keras.models import load_model
from scipy.spatial.distance import cosine


class FaceNetEncoder:
    def __init__(self, facenet_model_path=None, classifier_path=None):
        if facenet_model_path is not None and classifier_path is not None:
            self._base = FaceEncoder_FACENET(facenet_model_path, classifier_path)
        else:
            self._base = FaceEncoder_FACENET(classifier_path)

    def verify(self, sess, frame, bb, image_size, images_placeholder, phase_train_placeholder, embeddings, embedding_size):
        return self._base.verify(sess, frame, bb, image_size, images_placeholder, phase_train_placeholder, embeddings,
                                 embedding_size)

    def verify_h5(self, frame, face_bb):
        return self._base.verify_h5(frame, face_bb)

    def h5_recognizer(self, frame, face_bb):
        return self._base.h5_recognizer(frame, face_bb)

class FaceEncoder_FACENET:
    input_image_size = 160
    _face_crop_margin = 0

    cropped = []
    scaled = []
    scaled_reshape = []
    svc_model_classifier = None
    classifier_path = None

    def __init__(self, facenet_model_path=None, classifier_path=None):
        if facenet_model_path is not None and classifier_path is not None:
            # for h5
            if classifier_path.__contains__('officials_encodings.pkl'):
                self.facenet_model = load_model(facenet_model_path)
                classifier_filename_exp = os.path.expanduser(classifier_path)
                self.encoding_dict = facenet.load_pickle(classifier_filename_exp)
            else:
                self.facenet_model = load_model(facenet_model_path)
                classifier_filename_exp = os.path.expanduser(classifier_path)
                mm = joblib.load(classifier_filename_exp)
                (self.svc_model_classifier, _) = mm
        else:
            # for pb
            classifier_path = facenet_model_path
            print('loading... ', classifier_path)
            classifier_filename_exp = os.path.expanduser(classifier_path)
            mm = joblib.load(classifier_filename_exp)
            (self.svc_model_classifier, _) = mm

    def verify(self, sess, frame, bb, image_size, images_placeholder, phase_train_placeholder, embeddings, embedding_size):
        emb_array = np.zeros((1, embedding_size))
        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
        cropped = facenet.flip(cropped, False)
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, self.input_image_size, self.input_image_size, 3)
        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
        predictions = self.svc_model_classifier.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        print('best_class_indices= ', best_class_indices[0], '\tconfidence= ', best_class_probabilities)

        if best_class_probabilities > 0.20:
            return best_class_indices[0], best_class_probabilities[0]
        else:
            return "Unknown", best_class_probabilities[0]

    def verify_h5(self, frame, face_bb):
        cropped = frame[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2], :]
        cropped = facenet.flip(cropped, False)
        scaled = misc.imresize(cropped, (182, 182), interp='bilinear')
        scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)

        face_emb = facenet.get_embedding(self.facenet_model, scaled)
        predicted_class, prediction_prob = facenet.get_encode(self.svc_model_classifier, face_emb)

        if prediction_prob[0] > 0.15:
            return predicted_class, prediction_prob[0]
        else:
            return "Unknown", prediction_prob[0]

    def h5_recognizer(self, frame, face_bb):
        face, pt_1, pt_2 = facenet.get_face(frame, face_bb)
        encode = facenet.get_encode_2(self.facenet_model, face, (160, 160))
        encode = facenet.l2_normalizer.transform(encode.reshape(1, -1))[0]
        id = 'Unknown'

        distance = float("inf")
        for customer_id, db_encode in self.encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < 0.5 and dist < distance:
                id = customer_id
                distance = dist

        print('id= ', id, '\tdist= ', distance)
        return id, distance
