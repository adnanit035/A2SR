import numpy as np
import os
import facenet
import imageio
from a2sr_liveness_and_recognition import face_detect_and_align
from PIL import Image

def crop_and_align_dataset(input_dir_dataset, output_dir_path, mtcnn, image_size):
    # Add a random key to the filename to allow alignment using multiple processes

    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = facenet.get_dataset(input_dir_dataset)

    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.jpg')
                if not os.path.exists(output_filename):
                    try:
                        img = imageio.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        face_patches, padded_bounding_boxes, landmarks = face_detect_and_align.detect_faces(
                            img, mtcnn)

                        if len(padded_bounding_boxes) > 1:
                            print('More than 1 faces are detected...Unable to crop and align "%s"' % image_path)
                            continue

                        if len(face_patches) > 0:
                            for bb, landmark in zip(padded_bounding_boxes, landmarks):
                                print(image_path)
                                bb_temp = np.zeros(4, dtype=np.int32)
                                bb_temp[0] = bb[0]
                                bb_temp[1] = bb[1]
                                bb_temp[2] = bb[2]
                                bb_temp[3] = bb[3]

                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]

                                try:
                                    cropped_temp = Image.fromarray(cropped_temp)
                                    scaled_temp = cropped_temp.resize((image_size, image_size), Image.ANTIALIAS)
                                    nrof_successfully_aligned += 1
                                    imageio.imsave(output_filename, scaled_temp)
                                    text_file.write('%s %d %d %d %d\n' % (
                                        output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                                except:
                                    pass
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    return nrof_images_total, nrof_successfully_aligned


# input_dir_dataset = r'C:/Users/adnan/PycharmProjects/A2SR_Updated/face_recognition/dataset_preparation/tevta/'
# output_dir_path = r'C:/Users/adnan/PycharmProjects/A2SR_Updated/face_recognition/dataset_preparation/tevta_cropped/'
# input_dir_model_face_detection = r"C:/Users/adnan/PycharmProjects/A2SR_Updated/face_detection/detection/mtcnn_weights"
#
# import tensorflow as tf
#
# with tf.Graph().as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     with sess.as_default():
#         # create and initialize object of MTCNN for face detection
#         mtcnn = face_detect_and_align.create_mtcnn(sess, input_dir_model_face_detection)
#         image_size = 182
#
#         crop_and_align_dataset(input_dir_dataset, output_dir_path, mtcnn, image_size)
