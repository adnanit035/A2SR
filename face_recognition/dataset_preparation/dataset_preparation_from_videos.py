import os
import cv2

def crop_and_align_dataset(input_dir_dataset, output_dir_path):
    # Add a random key to the filename to allow alignment using multiple processes

    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = os.path.expanduser(input_dir_dataset)

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(r'C:\Users\Adnan\PycharmProjects\RAHBAR-Updated\face_recognition\dataset_preparation\new'):
        for file in f:
            if '.mkv' in file:
                files.append(os.path.join(r, file))

    for f in files:
        # print(f)
        cls_name = f.split('/')[-1].split('.')[0]
        cls_dir = os.path.join(output_dir, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

            capture = cv2.VideoCapture(f)
            frame_count = 0
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 2 == 1:
                    name = '{0}.jpg'.format(frame_count)
                    name = os.path.join(cls_dir, name)
                    # img_flip_lr = cv2.flip(frame, -1)
                    # img_rotate_90_clockwise = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(name, frame)
                else:
                    pass

            print(cls_name + ' => Done.')


input_dir_dataset = r'C:\Users\adnan\PycharmProjects\A2SR_Updated\face_recognition\dataset_preparation\new'
output_dir_path = './new_uni_teachers_frames/'
crop_and_align_dataset(input_dir_dataset, output_dir_path)
