import numpy as np
import cv2
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
from scipy.spatial import distance
from joblib import dump,load
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
class emotion_recog:
    def __init__(self,train_imgs_arr,test_imgs_arr,train_label,test_label):
        self.train_imgs_arr=train_imgs_arr
        self.test_imgs_arr=test_imgs_arr
        self.train_label=train_label
        self.test_label=test_label

    def get_vocab(self,):
        if os.path.exists('vocab.joblib'):
            vocab=load('vocab.joblib')
        else:
            total_features =None
            for i in range(len(self.train_imgs_arr)):
                sift=cv2.SIFT_create()
                keypoints,descriptors=sift.detectAndCompute(self.train_imgs_arr[i],None)

                # Some images might not have any keypoints detected or dimension might not be 128
                if descriptors is not None and descriptors.shape[1]==128:
                    if total_features is None:
                        total_features=descriptors
                    else:
                        total_features=np.vstack((total_features,descriptors))

            vocab_size=20
            kmeans=KMeans(n_clusters=vocab_size,init='random',n_init="auto",)
            kmeans.fit(total_features)
            vocab=kmeans.cluster_centers_
            dump(vocab,'vocab.pkl')
        return vocab

    def create_hist(self):
        vocab=self.get_vocab()
        if os.path.exists('train_hist_list.npy'):
            train_hist_list=np.load('train_hist_list.npy')
            test_hist_list=np.load('test_hist_list.npy')
            train_label_list=np.load('train_label_list.npy')
            test_label_list=np.load('test_label_list.npy')
        else:
            train_hist_list=[]
            test_hist_list=[]
            train_label_list=[]
            test_label_list=[]
            count=0
            for i in range(len(self.train_imgs_arr)):
                sift = cv2.SIFT_create()
                descriptors = sift.detectAndCompute(self.train_imgs_arr[i], None)[1]
                if descriptors is not None:
                    indices=np.argmin(distance.cdist(descriptors,vocab),axis=1)
                    hist,_=np.histogram(indices,bins=len(vocab),density=True)
                    train_hist_list.append(hist)
                    train_label_list.append(self.train_label[i])
            train_hist_list=np.array(train_hist_list)
            train_label_list=np.array(train_label_list)

            for j in range(len(self.test_imgs_arr)):
                sift = cv2.SIFT_create()
                test_descriptors = sift.detectAndCompute(self.test_imgs_arr[j], None)[1]
                if test_descriptors is not None:
                    test_indices = np.argmin(distance.cdist(test_descriptors, vocab), axis=1)
                    hist, _ = np.histogram(test_indices, bins=len(vocab), density=True)
                    test_hist_list.append(hist)
                    test_label_list.append(self.test_label[j])
            test_hist_list=np.array(test_hist_list)
            test_label_list=np.array(test_label_list)
            np.save('train_hist_list.npy',train_hist_list)
            np.save('test_hist_list.npy', test_hist_list)
            np.save('train_label_list.npy', train_label_list)
            np.save('test_label_list.npy', test_label_list)

        return train_hist_list,train_label_list,test_hist_list,test_label_list

    def svm_classification(self):
        class_dic={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
        train_hist_list,train_label_list,test_hist_list,test_label_list=self.create_hist()
        if os.path.exists('svm_best_model.joblib'):
            best_model=load('svm_best_model.joblib')
            best_accuracy=np.load('best_accuracy_cross_valid.npy')
            best_params=np.load('best_params.npy')
        else:
            # List of parameters to tune and test on
            C = [0.1, 1, 10]
            kernel = ['linear', 'rbf', 'sigmoid', 'poly']
            deg = [2, 3, 4]
            best_accuracy = 0
            best_model = None
            best_params = {}
            for C_val in C:
                for k in kernel:
                        if k=='poly':
                            for deg_val in deg:
                                model=svm.SVC(C=C_val,kernel=k,degree=deg_val)
                                scores = cross_val_score(model, train_hist_list, train_label_list, cv=4)
                                avg_accuracy = np.mean(scores)
                                print(f'Average train accuracy with {k} kernel, degree {deg_val}, regularizing coeff {C_val}: {avg_accuracy * 100: .4f}%')
                                if avg_accuracy > best_accuracy:
                                    best_model = model
                                    best_accuracy = avg_accuracy
                                    best_params['C'] = C_val
                                    best_params['kernel'] = k
                                    best_params['degree'] = deg_val

                        else:
                            model=svm.SVC(C=C_val,kernel=k)
                            scores=cross_val_score(model,train_hist_list,train_label_list,cv=4)
                            avg_accuracy=np.mean(scores)
                            print(f'Average train accuracy with {k} kernel, regularizing coeff {C_val}: {avg_accuracy * 100: .4f}%')
                            if avg_accuracy>best_accuracy:
                                best_model=model
                                best_accuracy=avg_accuracy
                                best_params['C']=C_val
                                best_params['kernel']=k
            np.save('best_accuracy_cross_valid.npy', best_accuracy)
            np.save('best_params.npy',best_params)
            dump(best_model, 'svm_best_model.joblib')
        print('----')
        print(f'Best model:{best_model},best train accuracy while validation: {best_accuracy * 100: .4f}%, best parameters= {best_params}')


        #Training on the whole data
        best_model.fit(train_hist_list,train_label_list)
        train_pred=best_model.predict(train_hist_list)
        count=0
        img_loc=[]
        for i in range(len(train_hist_list)):
            if train_pred[i]==train_label_list[i]:
                img_loc.append([i,train_pred[i]])
                count+=1
        train_acc = count / len(train_label_list)
        print(f'Training accuracy on whole training data:{train_acc*100: .4f}%')
        train_per_class_acc=precision_score(train_label_list,train_pred,average=None,zero_division=0)
        print(f'Training accuracy per class: {class_dic[0]}:{train_per_class_acc[0]*100:.4f}% | {class_dic[1]}:{train_per_class_acc[1]*100:.4f}% | {class_dic[2]}:{train_per_class_acc[2]*100:.4f}% | {class_dic[3]}:{train_per_class_acc[3]*100:.4f}% | {class_dic[4]}:{train_per_class_acc[4]*100:.4f}% '
        f'{class_dic[5]}: {train_per_class_acc[5] * 100: .4f}% | {class_dic[6]}:{train_per_class_acc[6]*100:.4f}%')

        # Testing on whole data
        test_pred=best_model.predict(test_hist_list)
        test_img_loc=[]
        test_count=0
        for i in range(len(test_hist_list)):
            if test_pred[i]==test_label_list[i]:
                test_img_loc.append([i, test_pred[i]])
                test_count+=1

        test_acc=test_count/len(test_label_list)
        print(f'Testing accuracy:{test_acc*100: .4f}%')
        test_per_class_acc = precision_score(test_label_list, test_pred, average=None, zero_division=0)
        print(f'Testing accuracy per class: {class_dic[0]}:{test_per_class_acc[0] * 100:.4f}% | {class_dic[1]}:{test_per_class_acc[1] * 100:.4f}% | {class_dic[2]}:{test_per_class_acc[2] * 100:.4f}% | {class_dic[3]}:{test_per_class_acc[3] * 100:.4f}% | {class_dic[4]}:{test_per_class_acc[4] * 100:.4f}% '
            f'{class_dic[5]}: {test_per_class_acc[5] * 100: .4f}% | {class_dic[6]}:{test_per_class_acc[6] * 100:.4f}%')

if __name__=='__main__':
    if os.path.exists('train_imgs_arr.npy'):
        train_imgs_arr=np.load('train_imgs_arr.npy')
        test_imgs_arr=np.load('test_imgs_arr.npy')
    else:
        train_imgs_arr=[]
        train_img_path = r'C:/Users/udayr/PycharmProjects/MLfiles/Project_data/All_train_images'
        train_seq_files = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
        train_seq_numbers = [int(f.split('_')[1].split('.')[0]) for f in train_seq_files]
        train_seq_files=[f for _, f in sorted(zip(train_seq_numbers, train_seq_files))]
        for train_img_name in train_seq_files:
            train_img = cv2.imread(train_img_path + '\\' + train_img_name,cv2.IMREAD_GRAYSCALE)
            train_imgs_arr.append(np.asarray(train_img))
        test_imgs_arr=[]
        test_img_path = r'C:/Users/udayr/PycharmProjects/MLfiles/Project_data/All_test_images'
        test_seq_files = [f for f in os.listdir(test_img_path) if f.endswith('.jpg')]
        test_seq_numbers = [int(f.split('_')[1].split('.')[0]) for f in test_seq_files]
        test_seq_files = [f for _, f in sorted(zip(test_seq_numbers, test_seq_files))]
        for test_img_name in test_seq_files:
            test_img = cv2.imread(test_img_path + '\\' + test_img_name,cv2.IMREAD_GRAYSCALE)
            test_imgs_arr.append(np.asarray(test_img))
        np.save('train_imgs_arr.npy',train_imgs_arr)
        np.save('test_imgs_arr.npy',test_imgs_arr)

    train_df = pd.read_csv('C:/Users/udayr/PycharmProjects/MLfiles/Project_data/train.csv')
    train_label=train_df['emotion']
    test_df=pd.read_csv('C:/Users/udayr/PycharmProjects/MLfiles/Project_data/icml_face_data.csv')
    public_test=test_df[test_df['Usage'] == 'PublicTest'].reset_index(drop=True)
    pvt_test=test_df[test_df['Usage'] == 'PrivateTest']
    pvt_test.reset_index(drop=True, inplace=True)
    pvt_test.index = pvt_test.index + len(public_test)
    test_data=pd.concat([public_test, pvt_test])
    test_label=test_data['emotion']
    er=emotion_recog(train_imgs_arr,test_imgs_arr,train_label,test_label).svm_classification()


