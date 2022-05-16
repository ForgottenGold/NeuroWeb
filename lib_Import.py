print("Импорт библиотек...")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as mpl
import tensorflow as tf
import random as rand
import pickle as pic
import numpy as np
import pylab as pl
import cv2
import os

from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPool1D, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from numpy import array, argmax, arange
