import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def mean_per_class_acc(num_of_classes, model_prediction, actual_prediction):
    ind = 0
    correct_pred = [0] * num_of_classes
    total_inst = [0] * num_of_classes
    while ind < actual_prediction.shape[0]:
        if model_prediction[ind] == actual_prediction[ind]:
            correct_pred[model_prediction[ind]] += 1
        total_inst[actual_prediction[ind]] += 1
        ind += 1
    return np.array(correct_pred), np.array(total_inst)

def get_features(data_path):
    image_classes = os.listdir(data_path)
    labels = []
    features = np.empty((0,0))
    for image_class in image_classes:
        class_path = os.path.join(data_path, image_class)    
        for data in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, data), cv2.IMREAD_GRAYSCALE)
            img = np.array(img).astype(np.float32).reshape((1, -1))
            img = (img * 2 / 255) - 1
            if features.any() :
                features = np.vstack((features, img))
            else:
                features = img
            labels.append(image_class)
    return features, labels

if __name__ == "__main__":
    train_data_path = "D:\\NEU\\3 Sem\ML\\Final Project\\archive\\train"
    test_data_path = "D:\\NEU\\3 Sem\ML\\Final Project\\archive\\test"

    classes = os.listdir(train_data_path)
    train_features, train_labels = get_features(train_data_path)
    test_features, test_labels = get_features(test_data_path)
    onehot_train_label = LabelBinarizer().fit_transform(np.array(train_labels))
    onehot_test_label = LabelBinarizer().fit_transform(np.array(test_labels))
    X_train, y_train = shuffle(train_features, onehot_train_label, random_state=42)
    y_train = np.argmax(y_train, axis=1)
    LR = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', multi_class='multinomial', max_iter=10000)
    LR.fit(X_train, y_train)
    y_pred_trn = LR.predict(X_train)
    acc_trn = sklearn.metrics.accuracy_score(y_pred_trn, y_train)
    print(f"Logistic Regression accuracy for training = {acc_trn}")
    y_pred_tst = LR.predict(test_features)
    test_labels = np.argmax(onehot_test_label, axis=1)
    acc_tst = sklearn.metrics.accuracy_score(y_pred_tst, test_labels)
    print(f"Logistic Regression accuracy for test data = {acc_tst}")
    per_class_corr_tst, per_class_total_tst = mean_per_class_acc(num_of_classes=len(classes),
                                                                    model_prediction=y_pred_tst,
                                                                    actual_prediction=test_labels)
    per_class_corr_trn, per_class_total_trn = mean_per_class_acc(num_of_classes=len(classes),
                                                                    model_prediction=y_pred_trn,
                                                                    actual_prediction=y_train)

    acc_per_class_test = per_class_corr_tst * 100 / per_class_total_tst

    acc_per_class_trn = per_class_corr_trn * 100 / per_class_total_trn

    plt.bar(classes, acc_per_class_test)

    plt.title('Accuracy per class (test)')
    plt.xlabel('Class')
    plt.ylabel('Accuracy %')

    plt.show()