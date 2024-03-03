import math
import os
from sklearn.externals import joblib
import numpy as np
from sklearn.svm import SVC
import facenet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_face(dir):
    faces = list()
    required_size = (160, 160)

    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename

        # load image from file
        image = Image.open(path)

        # convert to RGB, if needed
        image = image.convert('RGB')

        # resize pixels to the model size
        image = image.resize(required_size)

        face_array = np.asarray(image)
        faces.append(face_array)
    return faces


def load_dataset(dir):
    # list for faces and labels
    cls_names = []
    X, Y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
        X.extend(faces)
        Y.extend(labels)
        cls_names.append(subdir)
    return np.asarray(X), np.asarray(Y), cls_names


def train_classifier_using_facenetH5(train_output_dir_path, test_output_dir_path, input_dir_classifier, classifier_name,
                                     facenet_keras_model):

    trainX, trainY, train_class_names = load_dataset(train_output_dir_path)
    print('Total Number of classes in Training Dataset: %d' % len(train_class_names))
    print('Total Number of images in Training Dataset: %d' % len(trainX))

    testX, testY, test_class_names = load_dataset(test_output_dir_path)
    print('Total Number of classes in Testing Dataset: %d' % len(test_class_names))
    print('Total Number of images in Testing Dataset: %d' % len(testX))

    if facenet_keras_model is not None:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import Normalizer

        print('Calculating features for training faces...')
        emb_trainX_array = calculate_features_h5(facenet_keras_model, trainX)

        print('Calculating features for testing faces...')
        emb_testX_array = calculate_features_h5(facenet_keras_model, testX)

        # normalize input vectors
        in_encoder = Normalizer()
        emdTrainX_norm = in_encoder.transform(emb_trainX_array)
        emdTestX_norm = in_encoder.transform(emb_testX_array)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainY)
        trainy_enc = out_encoder.transform(trainY)
        testy_enc = out_encoder.transform(testY)
        # fit model
        print('Training classifier...')
        model = SVC(kernel='linear', probability=True)
        model.fit(emdTrainX_norm, trainy_enc)

        classifier_filename_exp = os.path.expanduser(input_dir_classifier + 'keras_' + classifier_name)
        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            joblib.dump((model, train_class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

        # predict
        yhat_train = model.predict(emdTrainX_norm)
        yhat_test = model.predict(emdTestX_norm)

        # summarizing classifier performance results.
        # 1. score
        print()
        print('================================================================================')
        score_train = accuracy_score(trainy_enc, yhat_train)
        score_test = accuracy_score(testy_enc, yhat_test)
        print('--------------------------------------------------------------------------------')
        print("Dataset: train=%d, test=%d" % (emb_trainX_array.shape[0], emb_testX_array.shape[0]))
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
        print('--------------------------------------------------------------------------------')

        # 2. generate classification report
        print()
        print('--------------------------------------------------------------------------------')
        print('Classification Report')
        print()
        print(classification_report(testy_enc, yhat_test, target_names=test_class_names))
        print('--------------------------------------------------------------------------------')

        # 3. generate confusion matrix
        print()
        print('--------------------------------------------------------------------------------')
        mat = generate_confusion_matrix(testy_enc, yhat_test, test_class_names)
        # finding accuracy from the confusion matrix.
        a = mat.shape
        corrPred = 0
        falsePred = 0

        for row in range(a[0]):
            for c in range(a[1]):
                if row == c:
                    corrPred += mat[row, c]
                else:
                    falsePred += mat[row, c]
        print('Correct predictions: ', corrPred)
        print('False predictions: ', falsePred)
        print('Accuracy of the Classification: ', (corrPred / (mat.sum()) * 100))
        print('--------------------------------------------------------------------------------')
        print('================================================================================')


def train_classifier(train_output_dir_path, test_output_dir_path, input_dir_classifier, classifier_name, sess,
                     embeddings, embedding_size, images_placeholder, phase_train_placeholder, facenet_keras_model=None):
    train_dataset = facenet.get_dataset(train_output_dir_path)
    trainX, trainY = facenet.get_image_paths_and_labels(train_dataset)
    print('Total Number of classes in Training Dataset: %d' % len(train_dataset))
    print('Total Number of images in Training Dataset: %d' % len(trainX))

    test_dataset = facenet.get_dataset(test_output_dir_path)
    testX, testY = facenet.get_image_paths_and_labels(test_dataset)
    print('Total Number of classes in Testing Dataset: %d' % len(test_dataset))
    print('Total Number of images in Testing Dataset: %d' % len(testX))

    # Create a list of class names for training dataset
    class_names = [cls.name.replace('_', ' ') for cls in train_dataset]
    print('class_names= ', class_names)

    # Create a list of class names for testing dataset
    test_class_names = [cls.name.replace('_', ' ') for cls in test_dataset]
    print('test_class_names= ', test_class_names)

    # Run forward pass to calculate embeddings for training dataset
    print('Calculating features for training images...')
    emb_array = calculate_features(sess, embeddings, embedding_size, images_placeholder, phase_train_placeholder,
                                   trainX)

    # Run forward pass to calculate embeddings for testing dataset
    print('Calculating features for testing images...')
    test_emb_array = calculate_features(sess, embeddings, embedding_size, images_placeholder, phase_train_placeholder,
                                        testX)
    classifier_filename_exp = os.path.expanduser(input_dir_classifier + classifier_name)

    # train classifier
    print('Training classifier...')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, trainY)

    # Saving classifier model
    with open(classifier_filename_exp, 'wb') as outfile:
        joblib.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)
    print()

    # predict
    train_predictions = model.predict(emb_array)
    test_predictions = model.predict(test_emb_array)

    # summarizing classifier performance results.
    # 1. score
    print()
    print('================================================================================')
    score_train = accuracy_score(trainY, train_predictions)
    score_test = accuracy_score(testY, test_predictions)
    print('--------------------------------------------------------------------------------')
    print("Dataset: train=%d, test=%d" % (len(train_dataset), len(test_dataset)))
    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
    print('--------------------------------------------------------------------------------')

    # 2. generate classification report
    print()
    print('--------------------------------------------------------------------------------')
    print('Classification Report')
    print()
    print(classification_report(testY, test_predictions, target_names=test_class_names))
    print('--------------------------------------------------------------------------------')

    # 3. generate confusion matrix
    print()
    print('--------------------------------------------------------------------------------')
    mat = generate_confusion_matrix(testY, test_predictions, test_class_names)
    # finding accuracy from the confusion matrix.
    a = mat.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += mat[row, c]
            else:
                falsePred += mat[row, c]
    print('Correct predictions: ', corrPred)
    print('False predictions: ', falsePred)
    print('Accuracy of the Classification: ', (corrPred / (mat.sum()) * 100))
    print('--------------------------------------------------------------------------------')
    print('================================================================================')


def calculate_features_h5(keras_model, X):
    emd_X = list()
    for face in X:
        emd = facenet.get_embedding(keras_model, face)
        emd_X.append(emd)

    emd_X = np.asarray(emd_X)
    return emd_X


def calculate_features(sess, embeddings, embedding_size, images_placeholder, phase_train_placeholder, paths):
    batch_size = 1000
    image_size = 160
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    return emb_array


def generate_confusion_matrix(data_features, predicitons, class_names):
    sns.set()
    fig, ax = plt.subplots(figsize=(25, 25))  # Sample figsize in inches

    mat = confusion_matrix(data_features, predicitons)
    sns.heatmap(mat.T, square=True, annot=True, linewidths=.5, ax=ax, fmt='d', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    return mat
