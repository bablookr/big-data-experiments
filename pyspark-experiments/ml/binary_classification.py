import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, NaiveBayes

spark = SparkSession.builder.appName('binary_classification').getOrCreate()


def logistic_regression(df):
    lr = LogisticRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
    lr_model = lr.fit(df)
    weights = lr_model.coefficients, lr_model.intercept
    predictions = lr_model.transform(df)
    return lr_model, weights, predictions


if __name__ == '__main__':
    df = spark.createDataFrame([
        (1, Vectors.dense(0.0, 5.0)),
        (0, Vectors.dense(1.0, 2.0)),
        (1, Vectors.dense(2.0, 1.0)),
        (0, Vectors.dense(3.0, 3.0))], ['label', 'features'])

    lr_model, weights, predictions = logistic_regression(df)
    print('weights = ', weights)
    predictions.select('features', 'label', 'prediction').show()