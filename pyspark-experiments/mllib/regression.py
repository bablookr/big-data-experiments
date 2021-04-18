import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD

spark = SparkSession.builder.appName('regression').getOrCreate()
sc = spark.sparkContext


def linear_regression(train_rdd, test_rdd):
    lr_model = LinearRegressionWithSGD.train(train_rdd)
    predictions = lr_model.predict(test_rdd)
    return lr_model, predictions


if __name__ == '__main__':
    train_rdd = sc.parallelize([
        LabeledPoint(1.0, [0.0, 5.0]),
        LabeledPoint(2.0, [1.0, 2.0]),
        LabeledPoint(3.0, [2.0, 1.0]),
        LabeledPoint(4.0, [3.0, 3.0])])

    test_rdd = sc.parallelize([
        [0.0, 5.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    lr_model, predictions = linear_regression(train_rdd, test_rdd)
    print(predictions.collect())