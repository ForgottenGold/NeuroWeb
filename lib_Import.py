print("Импорт библиотек...")
import warnings
warnings.filterwarnings('always');
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
import pickle
import cv2
import os

from keras.optimizer_v2.gradient_descent import SGD
#from keras.optimizer_v2 import Adam
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Conv1D, Conv2D, MaxPool1D, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from numpy import array, argmax, arange