# Emotion-Recognition-from-Facial-Expressions
This repository contains the code and research paper for the project on Emotion Recognition from facial expressions. In this project, different machine learning models were explored to recognize emotions from facial expressions. The aim is to improve human-computer interaction and explore various approaches to achieve accurate emotion recognition.

## Dataset
1. https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data
This is in the form of csv
2. https://www.kaggle.com/datasets/msambare/fer2013
This is in the form of grayscale images

## Key features
- **Multi-Modal Approach**: Integrates SVM, Logistic Regression, and CNN for robust emotion recognition.
- **Diverse Feature Extraction**: Leverages Bag of Visual Words, pixel values, and CNN layers.
- **Optimized Performance**: Rigorous hyperparameter tuning enhances model accuracy.
- **Data Enhancement**: Preprocessing and augmentation ensure models learn from comprehensive data.
- **Comprehensive Evaluation**: Detailed analysis includes accuracy metrics, confusion matrices, and benchmarking.
- **Proprietary CNN Model**: Tailored VGG16 architecture achieves 66.9% accuracy on FER2013 dataset.

## Methodology
- Three classification approaches were implemented: Support Vector Machines, Logistic Regression, and Convolutional Neural Networks.
- For SVM, bag-of-visual-words and histograms of descriptors were utilized as features. Also, hyper-parameter tuning was performed to obtain the best-performing model using rbf kernel.
-  Logistic Regression leverages pixel normalization. Here, hyperparameter tuning was performed using GridSearchCV.
-   In CNN, a pre-trained VGG16 model was experimented and a new hybrid model was proposed. The hybrid CNN model achieved the highest accuracy of 66.9%, surpassing SVM and Logistic Regression.

## Results
- **CNN Model**: Accuracy-66.9%
- **Logistic Regression model**: accuracy-38.35%
- **SVM**: accuracy- 25.69%
