import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
import os
from keras.layers import Input, Embedding, multiply, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras import applications
from keras import layers
from keras import optimizers
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.initializers import glorot_normal
import random
from sklearn.model_selection import train_test_split
from keras.utils import generic_utils
from keras.optimizers import Adam
import matplotlib.patches as mpatches
import scikitplot as skplt


def preprocessing(data):

    data['Amount'] = np.log10(data['Amount'].values + 1)
    data['Time'] = (data['Time'].values/3600)
    df_fraud = data[data['Class'] == 1]

    return data, df_fraud


def split(data):
    target = 'Class'

    # Divide the training data into training (80%) and test (20%)
    df_train, df_test = train_test_split(data, train_size=0.8, random_state=42, stratify=data[target])

    # Reset the index
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

    x_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    x_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    return x_train, x_test, y_train, y_test


def boxplot_compare(df1, df2, c1, c2, gan_label, title):
  fig, ax = plt.subplots(figsize=(16,10))
  bp1 = df1.boxplot(color=c1, showfliers=False)
  bp2 = df2.boxplot(color=c2, showfliers=False)

  patch1 = mpatches.Patch(color=c1, label='Original')
  patch2 = mpatches.Patch(color=c2, label=gan_label)
  plt.legend(handles=[patch1, patch2], prop={'size': 16})
  ax.set_title(title)
  plt.xticks(rotation=90)
  plt.show()


def gen_data(generator, n_data):
    noise = np.random.normal(0, 1, size=(n_data, 32))
    gen = generator.predict(noise)

    return gen


def concatenate(x_train, y_train, gen):
    x_train_gen = np.concatenate((x_train, gen))
    y_gen = np.array(gen.shape[0] * [1])
    y_train_gen = np.concatenate((y_train, y_gen))

    return x_train_gen, y_train_gen


# plot confusion matrix
def plot_cm(y_test, y_pred, title):
    skplt.metrics.plot_confusion_matrix(y_test, y_pred,figsize=(8,8))
    plt.title('Confusion Matrix ' + title)
    plt.show()


def check_performance(y_test, y_pred):
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1 score: ', f1_score(y_test, y_pred))
    print('ROC AUC score: ',  roc_auc_score(y_test, y_pred))


