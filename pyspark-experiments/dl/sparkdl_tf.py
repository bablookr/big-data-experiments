import numpy as np
import os
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from sparkdl import TFImageTransformer


spark = SparkSession.builder.appName('binary_classification').getOrCreate()