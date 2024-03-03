import cv2
import os
from time import time
import concurrent.futures
from statistics import mode
from keras.models import load_model
from liveness_detection.liveness import FaceLivenessModels, FaceLiveness
from imutils.video import FPS
from face_recognition.facenet_encoding import FaceNetEncoder
from mask_detection.face_mask_detection import MaskDetector
from robot_voice_engine import voice_speak_engine
from tensorflow.keras.models import load_model
import facenet
from face_detection import face_detect_and_align
import tensorflow as tf
from threading import Thread
from face_recognition import dataset_classification

# from arduino_communication import get_distance, is_someone_near

WINDOW_NAME = INPUT_DIR_MODEL_FACE_DETECTION = INPUT_DIR_MODEL_LIVENESS_DETECTION = ""
INPUT_DIR_DATASET = INPUT_DIR_FACENET_MODEL = INPUT_DIR_MODEL_CLASSIFIER = ""
FACENET_MODEL_NAME = CLASSIFIER_NAME = ""

MASK_DETECTOR_PROTOTXT_PATH = r"mask_detection\models\face_detector\deploy.prototxt"
MASK_DETECTOR_WEIGHTS_PATH = r"mask_detection\models\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
MASK_DETECTION_MODEL = r"mask_detection\models\mask_detector\mask_detector.model"


# method to initialize the webcam
def cam_init(cam_index, width, height):
    cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


# method to set labels on frames when system need to detect mask.
def label_face_mask(frame, label, mask, withoutMask, startX, startY, endX, endY):
    # set green color if mask detect else color will be red for bounding box rectangle and label on frames.
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # include the probability in the label
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    # display the label and bounding box rectangle on the output frame
    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


# process_livenessdetection is supposed to run before face recognition
def process_livenessdetection(cam_index, cam_resolution):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # create and initialize the MTCNN object for face face detection.
            mtcnn = face_detect_and_align.create_mtcnn(sess, INPUT_DIR_MODEL_FACE_DETECTION + 'mtcnn_weights/')
            image_size = 182

            # tevta classifier dec 21
            # HumanNames = ['Aamer Aziz GM (procurement)', 'Abdul wasay', 'Adnan Irshad', 'Ahmad umar', 'Ahmed Saeed', 'Akhtar Abbas', 'Ali Salman', 'Ali sufyan', 'Amir Hussain', 'Amir Tufail', 'Arish Ali', 'Hussnain Imtiaz', 'Irfan Abdullah', 'Islam Akbar', 'Javeed', 'Kashif', 'Mehboob Hussnain', 'Muhammad Aman', 'Muhammad Mehboob', 'Muhammad Rashid', 'Muhammad Umar', 'Muhammad Usman', 'Muhammad Yaseen', 'Muhammad Zulifqar', 'Naeem', 'Naeem hassan', 'Naveed Iqbal', 'Rao rashid', 'Rashid Muhammad', 'Rehan ahmed', 'Shams ijaz', 'Sher ali', 'Tanveer', 'Umar Javed', 'Waheed Zafar', 'Waheed zafar (DGM)', 'Waleed jahangir']
            # HumanNames = ['Aamer Aziz', 'Abdul wasay', 'Adnan Irshad', 'Ahmad Umar', 'Ahmed Saeed', 'Akhtar Abbas',
            #               'Ali Salman', 'Ali sufyan', 'Amir Hussain', 'Amir Tufail', 'Arish Ali', 'Aslam Iqbal',
            #               'Hussnain Imtiaz', 'Irfan Abdullah', 'Islam Akbar', 'Javeed Ahmad', 'Kashif Ahmad',
            #               'Mehboob Hussnain',
            #               'Mohammad Sarwar', 'Muhammad Aman', 'Muhammad Mehboob', 'Muhammad Rashid', 'Muhammad Umar',
            #               'Muhammad Usman', 'Muhammad Yaseen', 'Muhammad Zulifqar', 'Naeem Ahmad', 'Naeem hassan',
            #               'Naveed Iqbal',
            #               'Rao rashid', 'Rashid Muhammad', 'Rehan ahmed', 'Saad Sheikh', 'Shams Ijaz', 'Sher Ali',
            #               'Tanveer Ahmad',
            #               'Umar Javed', 'Waheed Zafar', 'Waheed zafar', 'Waleed jahangir'
            #               ]

            HumanNames = ['Abdul Aziz', 'Adeel', 'Adeel Abid', 'Adnan', 'Ahmad', 'Ameer Hamza', 'Ammar', 'Ans', 'Arish', 'Arsalan', 'Farhan Magsi', 'Hamza', 'Hamza Ashraf', 'Hamza Shoukat', 'Hamza Zafar', 'Jamshaid', 'Kamran', 'Mateen Ahmed', 'Mohsin', 'Mubashir', 'Mubashir Afzal', 'Muhammad Afzal', 'Muhammad Sarwar', 'Muhammad Shehzad', 'Muzzamil', 'Naveed Ahmad', 'Rizwan', 'Saad Tariq', 'Salman Qadir', 'Shafqat Mahmood', 'Shan', 'Shareef', 'Sheikh Burhan', 'Sikandar', 'Taha', 'Tanveer', 'Usama', 'VC', 'Waseem', 'Zeeshan', 'danial']
            # HumanNames = ['Abdul Aziz', 'Adeel', 'Adeel Abid', 'Ahmad', 'Ameer Hamza', 'Ammar', 'Ans', 'Arish', 'Arsalan', 'Farhan Magsi', 'Hamza', 'Hamza Ashraf', 'Hamza Shoukat', 'Hamza Zafar', 'Jamshaid', 'Kamran', 'Mateen Ahmed', 'Mohsin', 'Mubashir', 'Mubashir Afzal', 'Muhammad Afzal', 'Muhammad Sarwar', 'Muhammad Shehzad', 'Muzzamil', 'Naveed Ahmad', 'Rizwan', 'Saad Tariq', 'Salman Qadir', 'Shan', 'Shareef', 'Sheikh Burhan', 'Sikandar', 'Taha', 'Tanveer', 'Usama', 'Waseem', 'Zeeshan', 'danial']

            # Check facenet model for features extractions and classifier.
            if FACENET_MODEL_NAME.__contains__('.h5') and (
                    CLASSIFIER_NAME.__contains__('keras_') or CLASSIFIER_NAME.__contains__('officials_')):
                # create and initialize the FaceNetEncoder object using h5 pretrained-model for feature extraction
                facenet_encoder = FaceNetEncoder(facenet_model_path=INPUT_DIR_FACENET_MODEL + FACENET_MODEL_NAME,
                                                 classifier_path=INPUT_DIR_MODEL_CLASSIFIER + CLASSIFIER_NAME)
            elif FACENET_MODEL_NAME.__contains__('.pb') and not CLASSIFIER_NAME.__contains__('keras_'):
                print('Loading FaceNet Feature Extraction Model... ', INPUT_DIR_FACENET_MODEL + FACENET_MODEL_NAME)
                facenet.load_model(INPUT_DIR_FACENET_MODEL + FACENET_MODEL_NAME)
                # create and initialize the FaceNetEncoder object using pb pretrained-model for feature extraction
                facenet_encoder = FaceNetEncoder(classifier_path=INPUT_DIR_MODEL_CLASSIFIER + CLASSIFIER_NAME)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

            isPersonLive = isAuthorizedPerson = isUnauthorizedPerson = isMasked = False
            prevTime = frame_counter_for_liveness = frame_count_after_access = eyes_fake_face_counter = colorspace_fake_face_counter = eyes_real_face_counter = colorspace_real_face_counter = 0
            say_its_spoofing_attack = say_to_wear_mask_counter = say_to_welcome_counter = -1
            face_id, confidence = (None, 0)
            identified_person_name = ""

            # Initialize the camera
            camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])
            fps = FPS().start()

            # Create and Initialize object for face liveness detection
            # besed on eyes blinks and mouth open method
            face_liveness = FaceLiveness(model=FaceLivenessModels.EYESBLINK,
                                         path=INPUT_DIR_MODEL_LIVENESS_DETECTION)

            # Create and Initialize object for face liveness detection besed on colorspace method
            face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV,
                                          path=INPUT_DIR_MODEL_LIVENESS_DETECTION)

            # Create and Initialize object of Robot voice speak engine for speaking
            voice_engine = voice_speak_engine.VoiceSpeachEngine()

            # Loading model mask detection model weights using opencv dnn
            face_mask_detector = cv2.dnn.readNet(MASK_DETECTOR_PROTOTXT_PATH, MASK_DETECTOR_WEIGHTS_PATH)
            # load the face mask detector model
            mask_detector_model = load_model(MASK_DETECTION_MODEL)
            # Create and Initialize the object for mask detection
            face_mask_detector = MaskDetector(face_mask_detector, mask_detector_model)

            five_frames_predictions = []
            while True:
                # Capture frame from webcam
                ret, frame = camera.read()
                if frame is None:
                    print("Error, Check if camera is connected!")
                    break

                # caluclate time between each frames to determine Frame Rate
                curTime = time()
                sec = curTime - prevTime
                prevTime = curTime

                # detect person face, face landmarks and face bounding boxes from frame.
                face_patches, padded_bounding_boxes, landmarks = face_detect_and_align.detect_faces(frame, mtcnn)

                # if frame contains face then go for liveness detection

                # print(get_distance())
                # print(is_someone_near())

                if len(face_patches) > 0:  # and is_someone_near():
                    for bb, landmark in zip(padded_bounding_boxes, landmarks):
                        face_bb = tuple([bb[0], bb[1], bb[2], bb[3]])
                        text_x = bb[0]
                        text_y = bb[3] + 20

                        # thread for checking liveness through eyes blinks detection and determine mouth movements
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            eyes_close_thread = executor.submit(face_liveness.is_eyes_close_true, frame, face_bb)
                            eyes_close, eyes_ratio = eyes_close_thread.result()
                            print('eyes_close=', eyes_close, '\t==>Eyes_ratio=', eyes_ratio)

                        # thread for checking liveness, if frame is a print attack or replay attack based on colorspace
                        with concurrent.futures.ThreadPoolExecutor() as executor1:
                            is_fake_thread = executor1.submit(face_liveness2.is_fake_print_fake_replay, frame, face_bb)
                            is_fake_print, is_fake_replay = is_fake_thread.result()

                        # if not eyes_close and not isAuthorizedPerson and isPersonLive:
                        #     eyes_real_face_counter += 1
                        # else:
                        #     eyes_fake_face_counter += 1
                        #     if eyes_fake_face_counter >= 10:
                        #         face_id, confidence = ("Spoofing_Attack", None)
                        #         cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        #         cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #                     1, (0, 0, 255), thickness=1, lineType=2)
                        #
                        #         if say_its_spoofing_attack == -1:
                        #             say_its_spoofing_attack = 1
                        #             alert_thread = Thread(
                        #                 target=voice_engine.say_alert_for_spoofing_attack_female_voice,
                        #                 args=('robot_voice_engine/messages/alert.mp3',))
                        #             alert_thread.deamon = True
                        #             alert_thread.start()
                        #
                        #         frame_counter_for_liveness += 1
                        #         if frame_counter_for_liveness >= 10:
                        #             eyes_real_face_counter = eyes_fake_face_counter = 0
                        #             colorspace_real_face_counter = colorspace_fake_face_counter = 0
                        #             isPersonLive = False
                        #             say_its_spoofing_attack = -1
                        #
                        # if eyes_real_face_counter >= 3:
                        #     eyes_fake_face_counter = 0
                        #     if is_fake_print:
                        #         colorspace_fake_face_counter += 1
                        #         if colorspace_fake_face_counter >= 2:
                        #             face_id, confidence = ("Spoofing_Attack", None)
                        #             cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        #             cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #                         1, (0, 0, 255), thickness=1, lineType=2)
                        #             # print('Alert...Alert...Alert...its seems a spoofing attack')
                        #         if say_its_spoofing_attack == -1:
                        #             say_its_spoofing_attack = 1
                        #
                        #             alert_thread = Thread(
                        #                 target=voice_engine.say_alert_for_spoofing_attack_female_voice,
                        #                 args=('robot_voice_engine/messages/alert.mp3',))
                        #             alert_thread.deamon = True
                        #             alert_thread.start()
                        #
                        #         frame_counter_for_liveness += 1
                        #         if frame_counter_for_liveness >= 10:
                        #             eyes_real_face_counter = eyes_fake_face_counter = 0
                        #             colorspace_real_face_counter = colorspace_fake_face_counter = 0
                        #             isPersonLive = False
                        #             say_its_spoofing_attack = -1
                        #     else:
                        #         colorspace_real_face_counter += 1
                        #
                        #     if colorspace_real_face_counter >= 3:
                        #         colorspace_fake_face_counter = 0
                        #         isPersonLive = True

                        # Identify face only if it is not fake and eyes are open and mouth is moving
                        # if (is_fake_print or is_fake_replay) and not isAuthorizedPerson:
                        #     fake_face_counter += 1
                        #     if is_fake_print:
                        #         face_id, confidence = ("Fake", None)
                        #     if is_fake_replay:
                        #         face_id, confidence = ("Fake", None)

                        # cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        # cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #             1, (0, 0, 255), thickness=1, lineType=2)

                        # if (is_fake_print or is_fake_replay) and not isAuthorizedPerson:
                        #     colorspace_fake_face_counter += 1
                        #
                        #     face_id, confidence = ("Spoofing_Attack", None)
                        #     cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        #     cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #                 1, (0, 0, 255), thickness=1, lineType=2)
                        if eyes_close:
                            # if not eyes_close and not mouth_open:
                            eyes_real_face_counter += 1
                            print('eyes_real_face_counter= ', eyes_real_face_counter)
                            if eyes_real_face_counter >= 3:
                                print('real for rognition')
                                # if isPersonLive:
                                # Predict the the real detected person
                                if FACENET_MODEL_NAME.__contains__('.h5') and (
                                        CLASSIFIER_NAME.__contains__('keras_') or CLASSIFIER_NAME.__contains__(
                                        'officials_')):
                                    if CLASSIFIER_NAME.__eq__('officials_encodings.pkl'):
                                        best_predicted_class, confidence = facenet_encoder.h5_recognizer(frame, face_bb)
                                        if best_predicted_class is not 'Unknown':
                                            for H_i in HumanNames:
                                                if H_i == HumanNames[int(best_predicted_class) - 1]:
                                                    face_id = identified_person_name = HumanNames[
                                                        int(best_predicted_class) - 1]
                                                    face_id = face_id + '-' + f'{(confidence * 100):.2f}%'

                                                    # save predictions results in list to later on determining best result.
                                                    five_frames_predictions.append(best_predicted_class)
                                                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                                    cv2.putText(frame, face_id, (text_x, text_y),
                                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                                                                thickness=1,
                                                                lineType=2)

                                        elif best_predicted_class is "Unknown":
                                            five_frames_predictions.append("Unknown")
                                            face_id = best_predicted_class
                                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                            cv2.putText(frame, face_id, (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                                            # isUnauthorizedPerson = True
                                    else:
                                        # predict the face using facenet feature extraction and verification
                                        best_predicted_class, confidence = facenet_encoder.verify_h5(frame, face_bb)

                                        if best_predicted_class is not "Unknown":
                                            for H_i in HumanNames:
                                                if H_i == HumanNames[best_predicted_class]:
                                                    face_id = identified_person_name = HumanNames[best_predicted_class]
                                                    face_id = face_id + '-' + f'{(confidence * 100):.2f}%'

                                                    # save predictions results in list to later on determining best result.
                                                    five_frames_predictions.append(best_predicted_class)
                                                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                                    cv2.putText(frame, face_id, (text_x, text_y),
                                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                                                                thickness=1,
                                                                lineType=2)

                                        elif best_predicted_class is "Unknown":
                                            five_frames_predictions.append("Unknown")
                                            face_id = best_predicted_class
                                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                            cv2.putText(frame, face_id, (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                                            # isUnauthorizedPerson = True

                                elif FACENET_MODEL_NAME.__contains__('.pb') and not CLASSIFIER_NAME.__contains__(
                                        'keras_') and not isMasked and not isAuthorizedPerson:
                                    print('in pb recognition')
                                    # predict the face using facenet feature extraction and verification
                                    best_predicted_class, confidence = facenet_encoder.verify(sess, frame, face_bb,
                                                                                              image_size,
                                                                                              images_placeholder,
                                                                                              phase_train_placeholder,
                                                                                              embeddings,
                                                                                              embedding_size)

                                    print('\n = > single frame predictions : best_predicted_class= ',
                                          best_predicted_class, '\t confidence= ', confidence, '\n')

                                    if best_predicted_class is not "Unknown":
                                        for H_i in HumanNames:
                                            if H_i == HumanNames[best_predicted_class]:
                                                face_id = identified_person_name = HumanNames[best_predicted_class]
                                                face_id = face_id + '-' + f'{100 - (confidence * 100):.2f}%'
                                                # save predictions results in list to later on determining best result.
                                                five_frames_predictions.append(best_predicted_class)
                                                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                                cv2.putText(frame, face_id, (text_x, text_y),
                                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1,
                                                            lineType=2)

                                    elif best_predicted_class is "Unknown":
                                        five_frames_predictions.append("Unknown")
                                        face_id = best_predicted_class
                                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                                        cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                                        # isUnauthorizedPerson = True

                        # if length of list of at least last 5 five frame's results greater than 5 or more
                        if len(five_frames_predictions) >= 5:
                            try:
                                # get most frequent prediction in last 5 or more processed frames.
                                most_frequent_prediction = mode(five_frames_predictions)
                                if most_frequent_prediction is not "Unknown" and not 0:
                                    isAuthorizedPerson = True
                                    # face_id, confidence = most_frequent_prediction, confidence
                                    five_frames_predictions.clear()
                                elif most_frequent_prediction is "Unknown" and not 0:
                                    isUnauthorizedPerson = True
                                    face_id = "Unknown"
                                    five_frames_predictions.clear()
                            except:
                                pass

                        # if length of list of at least 5 or more process frames equal to 10 then reset it.
                        # And mark prediction as 'Unknown'
                        if len(five_frames_predictions) >= 10:
                            isUnauthorizedPerson = True
                            face_id = "Unknown"
                            five_frames_predictions.clear()

                        # Show label if person is un-authorized
                        if isUnauthorizedPerson:
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                            cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)

                            voice_engine.say_not_allowed_female_voice('robot_voice_engine/messages/say_not_allowed.mp3')

                            isAuthorizedPerson = isUnauthorizedPerson = isMasked = False
                            frame_count_after_access = eyes_fake_face_counter = colorspace_fake_face_counter = eyes_real_face_counter = colorspace_real_face_counter = 0
                            identified_person_name = ""
                            say_to_welcome_counter = say_to_wear_mask_counter = -1
                            face_id, confidence = (None, 0)

                        # Show label if person is un-authorized and then go for mask detection procedure.
                        if isAuthorizedPerson:
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                            cv2.putText(frame, face_id, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255), thickness=1, lineType=2)

                            # detect faces in the frame and determine if they are wearing a face mask or not
                            locations, predictions = face_mask_detector.detect_face_mask(frame, cv2)
                            # with concurrent.futures.ThreadPoolExecutor() as executor2:
                            #     mask_detection_thread = executor2.submit(face_mask_detector.detect_face_mask,
                            #                                       frame, cv2)
                            #     print('In Mask detection thread')
                            #     locations, predictions = mask_detection_thread.result()

                            # if say_to_welcome_counter == -1:
                            #     say_to_welcome_counter = 1
                            #
                            #     # set thread for welcome call to person and access.
                            #     # t2 = Thread(target=voice_engine.say_welcome, args=(identified_person_name,))
                            #     t2 = Thread(target=voice_engine.say_welcome, args=(identified_person_name,))
                            #     t2.deamon = True
                            #     t2.start()
                            #
                            #     isMasked = True
                            #     frame_count_after_access = 5
                            #
                            # if frame_count_after_access > 0:
                            #     frame_count_after_access -= 1
                            #
                            # if frame_count_after_access == 1:
                            #     print('reset all parameters')
                            #     isAuthorizedPerson = isUnauthorizedPerson = isMasked = False
                            #     frame_count_after_access = fake_face_counter = real_face_counter = 0
                            #     identified_person_name = ""
                            #     say_to_welcome_counter = say_to_wear_mask_counter = -1
                            #     face_id, confidence = (None, 0)

                            if predictions is not 0:
                                # loop over the detected face locations and their corresponding locations
                                for (box, pred) in zip(locations, predictions):
                                    # unpack the bounding box and predictions
                                    (startX, startY, endX, endY) = box
                                    (mask, withoutMask) = pred

                                    # determine the class label and color we'll use to draw
                                    # the bounding box and text
                                    if mask > withoutMask:
                                        label = "Mask"
                                        if say_to_welcome_counter == -1:
                                            say_to_welcome_counter = 1

                                            # set thread for welcome call to person and access.
                                            # set thread for welcome call to person and access.
                                            if identified_person_name.__eq__('Ali Salman'):
                                                t = Thread(target=voice_engine.say_welcome_female_voice,
                                                           args=('robot_voice_engine/messages/msg_for_ali_salman.mp3',))
                                                t.deamon = True
                                                t.start()
                                                frame_count_after_access = 150

                                            else:
                                                t = Thread(target=voice_engine.say_welcome_female_voice,
                                                           args=(
                                                           'robot_voice_engine/messages/thank you for mask kfueit.mp3',))
                                                t.deamon = True
                                                t.start()
                                                frame_count_after_access = 80

                                            isMasked = True
                                            # frame_count_after_access = 5

                                        label_face_mask(frame, label, mask, withoutMask, startX, startY, endX, endY)
                                    else:
                                        label = "No Mask"
                                        label_face_mask(frame, label, mask, withoutMask, startX, startY, endX, endY)
                                        if say_to_wear_mask_counter == -1:
                                            say_to_wear_mask_counter = 1

                                            # set thread for wear mask warning call to detected person.
                                            # t = Thread(target=voice_engine.say_to_wear_mask, args=(identified_person_name,))
                                            # t.deamon = True
                                            # t.start()

                                            if identified_person_name.__eq__('Ahmad Umar'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_ahmad_ummer_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Ahmed Saeed'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_ahmad_saeed_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Akhtar Abbas'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_akhtar_abbas_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Ali Salman'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_ali_salman_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Aslam Iqbal'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_aslam_iqbal_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Muhammad Sarwar'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_sarwar_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            elif identified_person_name.__eq__('Saad Sheikh'):
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_mr_saad_sheikh_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()
                                            else:
                                                t1 = Thread(
                                                    target=voice_engine.say_hello_and_ask_for_facemask_female_voice,
                                                    args=(
                                                        'robot_voice_engine/messages/hello_sir_and_ask_for_mask.mp3',))
                                                t1.deamon = True
                                                t1.start()

                                    # label_face_mask(frame, label, mask, withoutMask, startX, startY, endX, endY)

                        # print eyes landmarks on face.
                        for j in range(2):
                            size = 5
                            left_eye = (int(landmarks[0][j]) - size, int(landmarks[0][j + 5]) - size)
                            right_eye = (int(landmarks[0][j]) + size, int(landmarks[0][j + 5]) + size)
                            cv2.rectangle(frame, left_eye, right_eye, (255, 0, 255), 2)
                        # -------------------------------------------------

                if frame_count_after_access >= 0:
                    print('frame after access= ', frame_count_after_access)
                    # label_face_mask(frame, label, mask, withoutMask, startX, startY, endX, endY)
                    frame_count_after_access -= 1

                if frame_count_after_access == 1:
                    print('reset all parameters')
                    isPersonLive = isAuthorizedPerson = isUnauthorizedPerson = isMasked = False
                    eyes_real_face_counter = colorspace_real_face_counter = eyes_fake_face_counter = colorspace_fake_face_counter = frame_count_after_access = 0
                    identified_person_name = ""
                    say_to_welcome_counter = say_to_wear_mask_counter = -1
                    face_id, confidence = (None, 0)

                # measure Frame rate
                try:
                    cam_fps = 1 / sec
                    webcam_fps = 'FPS: %2.3f' % cam_fps
                    text_fps_x = len(frame[0]) - 150
                    text_fps_y = 20
                    cv2.putText(frame, webcam_fps, (text_fps_x, text_fps_y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                except:
                    pass

                # Display updated frame
                cv2.imshow(WINDOW_NAME, frame)

                # Check for user actions
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    fps.stop()
                    break

                fps.update()

            # Release the camera
            camera.release()
            cv2.destroyAllWindows()


def run_training(input_dir_model_face_detection, input_dir_facenet_model,
                 input_dir_classifier, input_dir_dataset, output_dir_path, facenet_model_name, classifier_name):
    from face_recognition.dataset_preparation import dataset_preprocessing

    # train_input_dir_dataset = input_dir_dataset + 'train/'
    train_input_dir_dataset = input_dir_dataset
    # test_input_dir_dataset = input_dir_dataset + 'test/'
    # train_output_dir_path = output_dir_path + 'train/'
    train_output_dir_path = output_dir_path
    # test_output_dir_path = output_dir_path + 'test/'

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # create and initialize object of MTCNN for face detection
            input_dir_model_face_detection = 'C:/Users/adnan/PycharmProjects/A2SR_Updated/face_detection/detection/'
            mtcnn = face_detect_and_align.create_mtcnn(sess, input_dir_model_face_detection + 'mtcnn_weights/')
            image_size = 182

            # preprocess test dataset by detecting and cropping and aligning faces for training classifier.
            total_dataset_images, total_successfully_aligned_images = dataset_preprocessing.crop_and_align_dataset(
                train_input_dir_dataset, train_output_dir_path, mtcnn, image_size)
            print('Total number of train images: ', total_dataset_images)
            print('Number of successfully aligned images: ', total_successfully_aligned_images)

            # preprocess test dataset by detecting and cropping and aligning faces for testing classifier.
            # total_dataset_images, total_successfully_aligned_images = dataset_preprocessing.crop_and_align_dataset(
            #     test_input_dir_dataset, test_output_dir_path, mtcnn, image_size)
            # print('Total number of test images: ', total_dataset_images)
            # print('Number of successfully aligned images: ', total_successfully_aligned_images)

            # if facenet_model_name.__contains__('.h5'):
            #     print('Loading FaceNet Feature Extraction Model...')
            #     facenet_keras_model = load_model(input_dir_facenet_model + facenet_model_name)
            #
            #     # train classifier on preprocessed(aligned) dataset.
            #     dataset_classification.train_classifier_using_facenetH5(train_output_dir_path, test_output_dir_path,
            #                                                             input_dir_classifier, classifier_name,
            #                                                             facenet_keras_model)
            # else:
            #     print('Loading FaceNet Feature Extraction Model...')
            #     facenet.load_model(input_dir_facenet_model + facenet_model_name)
            #
            #     images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #     embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #     phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #     embedding_size = embeddings.get_shape()[1]
            #
            #     # train classifier on preprocessed(aligned) dataset.
            #     dataset_classification.train_classifier(train_output_dir_path, test_output_dir_path,
            #                                             input_dir_classifier, classifier_name, sess,
            #                                             embeddings, embedding_size, images_placeholder,
            #                                             phase_train_placeholder)


def run_liveness(window_name, input_dir_model_detection, input_dir_model_liveness,
                 input_dir_facenet_model, input_dir_classifier, input_dir_dataset, cam_index,
                 cam_resolution, facenet_model_name, classifier_name):
    global WINDOW_NAME, INPUT_DIR_MODEL_FACE_DETECTION, INPUT_DIR_MODEL_LIVENESS_DETECTION, INPUT_DIR_FACENET_MODEL
    global INPUT_DIR_MODEL_CLASSIFIER, INPUT_DIR_DATASET, FACENET_MODEL_NAME, CLASSIFIER_NAME

    WINDOW_NAME = window_name
    INPUT_DIR_MODEL_FACE_DETECTION = input_dir_model_detection
    INPUT_DIR_MODEL_LIVENESS_DETECTION = input_dir_model_liveness
    INPUT_DIR_FACENET_MODEL = input_dir_facenet_model
    INPUT_DIR_MODEL_CLASSIFIER = input_dir_classifier
    INPUT_DIR_DATASET = input_dir_dataset
    FACENET_MODEL_NAME = facenet_model_name
    CLASSIFIER_NAME = classifier_name

    process_livenessdetection(cam_index, cam_resolution)
