# IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
import argparse
import warnings

warnings.filterwarnings('ignore')


# DEFINING CONSTANTS
FEATURES_FILE_NAME = 'features.csv'
MODELS_DIR = 'models'
CONFUSION_MATRIX_DIR = 'confusion_matrix'
ACCURACY_PLOT_FILE_NAME = 'model_accuracies.png'


# FUNCTION DEFINITIONS
def check_output_dir():
    """
    Check if the output directories exist and delete them if they do.
    Create the output directories.
    """

    if os.path.exists(MODELS_DIR):
        os.system(f'rm -rf {MODELS_DIR}')
    os.makedirs(MODELS_DIR)

    if os.path.exists(CONFUSION_MATRIX_DIR):
        os.system(f'rm -rf {CONFUSION_MATRIX_DIR}')
    os.makedirs(CONFUSION_MATRIX_DIR)

    return


def load_data():
    """
    Load the features from the csv file and return the data in the form of x_data and y_data (features and labels).
    """

    data = pd.read_csv(FEATURES_FILE_NAME)
    data = data.drop('file_name', axis=1)       # dropping the file_name column

    x_data = data.drop('vowel', axis=1)
    y_data = data['vowel']

    return x_data, y_data


def process_data(x_data, y_data):
    """
    Preprocess the data by splitting it into training and testing data and scaling the features using StandardScaler.
    """
    
    # Perform label encoding on the target variable
    y_data = y_data.astype('category')
    class_names = y_data.cat.categories.to_list()
    y_data = y_data.cat.codes

    # Split the data into training and testing data in 80:20 ratio
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=45, stratify=y_data)

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, class_names


def train_classifier(x_train, y_train, classifier):
    """
    Function to train the classifier on the training data and return the trained model and the accuracy on the training data.
    """

    # define the classifier
    if classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier == 'dt':
        model = DecisionTreeClassifier()
    elif classifier == 'svm':
        model = SVC()
    elif classifier == 'gmm':
        model = GaussianMixture(n_components=3)

    # train the model
    model.fit(x_train, y_train)
    
    # predict on the training data and calculate the accuracy
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)

    return model, accuracy*100


def evaluate_classifier(model, x_test, y_test):
    """
    Function to evaluate the classifier on the test data and return the accuracy and confusion matrix.
    """

    # predict on the test data
    y_pred = model.predict(x_test)

    # calculate the accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    return accuracy*100, confusion


def save_confusion_matrix(confusion, model_name, class_names):
    """
    Function to save the confusion matrix as a heatmap.
    """
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt='d', annot_kws={'size': 18}, cbar_kws={'shrink': 0.8}, xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)

    plt.savefig(f'{CONFUSION_MATRIX_DIR}/{model_name}.png')
    plt.close()


def save_accuracy_plot(train_accuracy_log, test_accuracy_log, classifiers):
    """
    Function to plit and save the accuracy graph of the classifiers.
    """
    
    plt.figure(figsize=(10, 7))
    plt.plot(classifiers, train_accuracy_log, label='Train')
    plt.plot(classifiers, test_accuracy_log, label='Test')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(ACCURACY_PLOT_FILE_NAME)
    plt.close()


# MAIN FUNCTION
if __name__=='__main__':
    # parser
    parser = argparse.ArgumentParser(description='Train and evaluate a classifier on the extracted features')
    parser.add_argument('--classifier', type=str, default='all', choices=['knn', 'dt', 'svm', 'gmm', 'all'], help='Classifier to use')
    
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, filename=f'classification_logs.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    check_output_dir()

    # load and process data
    x_data, y_data = load_data()
    x_train, x_test, y_train, y_test, class_names = process_data(x_data, y_data)

    # check the classifier to use
    if args.classifier == 'all':
        classifiers = ['knn', 'dt', 'svm', 'gmm']
    else:
        classifiers = [args.classifier]

    # to store the accuracy of the classifiers
    train_accuracy_log = []
    test_accuracy_log = []

    # loop over the classifiers
    for classifier in classifiers:
        logging.info(f'Training {classifier} classifier...')
        
        # train and test the classifier
        model, train_accuracy = train_classifier(x_train, y_train, classifier)
        test_accuracy, confusion = evaluate_classifier(model, x_test, y_test)

        train_accuracy_log.append(train_accuracy)
        test_accuracy_log.append(test_accuracy)

        logging.info(f'Training accuracy: {train_accuracy:.2f}')
        logging.info(f'Test accuracy: {test_accuracy:.2f}\n')

        # save the model and confusion matrix
        joblib.dump(model, f'{MODELS_DIR}/{classifier}.pkl')    
        save_confusion_matrix(confusion, classifier, class_names)
    
    # save the accuracy plot to compare the classifiers
    save_accuracy_plot(train_accuracy_log, test_accuracy_log, classifiers)

    logging.info('Training complete!')
