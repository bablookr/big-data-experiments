import numpy as np
import os
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from sparkdl import KerasImageFileTransformer


spark = SparkSession.builder.appName('binary_classification').getOrCreate()


model = InceptionV3(weights='imagenet')
model.save('model-full.h5')

IMAGES_PATH = 'datasets/image_classifier/test/'


def preprocess_keras_inceptionV3(uri):
  image = img_to_array(load_img(uri, target_size=(299, 299)))
  image = np.expand_dims(image, axis=0)
  return preprocess_input(image)


transformer = KerasImageFileTransformer(inputCol='uri',
                                        outputCol='predictions',
                                        modelFile='model-full-tmp.h5',
                                        imageLoader=preprocess_keras_inceptionV3)


files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
uri_df = spark.createDataFrame(files, StringType()).toDF('uri')

predictions = transformer.transform(uri_df)
predictions.select('uri', 'predictions').show()


