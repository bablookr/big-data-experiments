import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor

spark = SparkSession.builder.appName('regression').getOrCreate()


def linear_regression(df):
    regression = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
    model = regression.fit(df)
    weights = model.coefficients, model.intercept
    predictions = model.transform(df)
    return model, weights, predictions


if __name__ == '__main__':
    df = spark.createDataFrame([
        (1.0, Vectors.dense(0.0, 5.0)),
        (2.0, Vectors.dense(1.0, 2.0)),
        (3.0, Vectors.dense(2.0, 1.0)),
        (4.0, Vectors.dense(3.0, 3.0))], ['label', 'features'])

    lr_model, weights, predictions = linear_regression(df)
    print('weights = ', weights)
    predictions.select('features', 'label', 'prediction').show()