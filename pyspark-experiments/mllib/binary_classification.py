import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

spark = SparkSession.builder.appName('binary_classification').getOrCreate()
sc = spark.sparkContext


def logistic_regression(train_rdd, test_rdd):
    lr_model = LogisticRegressionWithSGD.train(train_rdd)
    predictions = lr_model.predict(test_rdd)
    return lr_model, predictions


if __name__ == '__main__':
    train_rdd = sc.parallelize([
        LabeledPoint(1, [0.0, 5.0]),
        LabeledPoint(0, [1.0, 2.0]),
        LabeledPoint(1, [2.0, 1.0]),
        LabeledPoint(0, [3.0, 3.0])])

    test_rdd = sc.parallelize([
        [0.0, 5.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    lr_model, predictions = logistic_regression(train_rdd, test_rdd)
    print(predictions.collect())