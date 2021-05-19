#create newapp
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd



import tensorflow as tf
print(tf.__version__)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

# Create some tensors and perform an operation
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
import cv2
import os
import tqdm
import heapq
import datetime
import glob
import matplotlib.pyplot as plt
from absl import logging
logging.set_verbosity(logging.ERROR)
# Some modules to help with reading the UCF101 dataset.
import random
import tempfile
import ssl
from IPython import display


#Load Saved Model
PATH=os.getcwd()
K_I_512D= tf.keras.models.load_model(PATH+'/saved_model/NoDrop_NoDense_5sec_AR_kinetics+ImageNetweightsonly')
K_I_aug_20E= tf.keras.models.load_model(PATH+'/saved_model/augumented_5sec_AR_kinetics+ImageNetweightsonly')
K_3DL= tf.keras.models.load_model(PATH+'/saved_model/5sec_AR_kineticsweightsonly_noflatten_moredense')
