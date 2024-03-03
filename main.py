import argparse
from a2sr_liveness_and_recognition import run_liveness, run_training

# Set width and height of webcam
RESOLUTION_QVGA = (480, 450)
RESOLUTION_VGA = (640, 480)

# Set the window name for webcam
WINDOW_NAME = "A2SR Face Recognition System"

# Set the input directories
INPUT_DIR_MODEL_FACE_DETECTION = "face_detection/detection/"
INPUT_DIR_MODEL_LIVENESS_DETECTION = "liveness_detection/models/"
INPUT_DIR_MODEL_FACENET_ENCODING_MODELS = "face_recognition/models/facenet_models/"
INPUT_DIR_MODEL_CLASSIFIER = "./face_recognition/models/classifier/"
# INPUT_DIR_DATASET = "face_recognition/dataset_preparation/TEVTA Dataset/"
INPUT_DIR_DATASET = "face_recognition/dataset_preparation/new/"

# Set the Output directories
# OUTPUT_DIR_PROCESSED_DATASET_PATH = 'face_recognition/dataset_preparation/TEVTA Dataset/'
OUTPUT_DIR_PROCESSED_DATASET_PATH = 'face_recognition/dataset_preparation/new_uni_teachers_frames_cropped/'

# Set Model Names
FACENET_MODEL_NAME = None
CLASSIFIER_NAME = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", default=0, help="Set Webcam Index => 0: Default Cam | 1: External Cam")
    parser.add_argument("--FaceNet_Model", choices=['pb', 'h5'], default='pb',
                        help="Select Pre-Trained FaceNet Model for Features Extraction")
    parser.add_argument("--operation", choices=['recognition', 'training'], default='recognition',
                        help="Operation to Perform")
    args = parser.parse_args()

    cam_index = int(args.webcam)
    cam_resolution = RESOLUTION_QVGA

    if args.FaceNet_Model == "pb":
        INPUT_DIR_MODEL_FACENET_ENCODING_MODELS += '20180402-114759/'
        FACENET_MODEL_NAME = '20180402-114759.pb'
        # CLASSIFIER_NAME = 'tevta_classifier_20180402-114759_dec_21_Updated.pkl'
        CLASSIFIER_NAME = 'exhibition_classifier_final.pkl'
        # CLASSIFIER_NAME = 'custom_classifier_final.pkl'
    elif args.FaceNet_Model == "h5":
        FACENET_MODEL_NAME = 'facenet_keras.h5'
        CLASSIFIER_NAME = 'custom_classifier_final.pkl'

    if args.operation == "recognition":
        run_liveness(WINDOW_NAME, INPUT_DIR_MODEL_FACE_DETECTION, INPUT_DIR_MODEL_LIVENESS_DETECTION,
                     INPUT_DIR_MODEL_FACENET_ENCODING_MODELS, INPUT_DIR_MODEL_CLASSIFIER, INPUT_DIR_DATASET,
                     cam_index, cam_resolution, FACENET_MODEL_NAME, CLASSIFIER_NAME)

    elif args.operation == "training":
        run_training(INPUT_DIR_MODEL_FACE_DETECTION, INPUT_DIR_MODEL_FACENET_ENCODING_MODELS,
                     INPUT_DIR_MODEL_CLASSIFIER, INPUT_DIR_DATASET, OUTPUT_DIR_PROCESSED_DATASET_PATH,
                     FACENET_MODEL_NAME, CLASSIFIER_NAME)
    else:
        print('Invalid Arguments...')
