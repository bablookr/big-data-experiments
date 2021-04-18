from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName('test_sql').getOrCreate()


class TestPySparkSQL():
    def test_collect(self):
        data = [('Alice', 'F', 1), ('Bob', 'M', 2), ('Cate', 'F', 3)]
        df = spark.createDataFrame(data, ['name', 'gender', 'age'])
        #df.show()
        rows = df.collect()
        row_1 = rows[0]
        assert row_1['name'] == 'Alice'
        assert row_1['gender'] == 'F'
        assert row_1['age'] == 1

    def test_take(self):
        data = [('Alice', 'F', 1), ('Bob', 'M', 2), ('Cate', 'F', 3)]
        df = spark.createDataFrame(data, ['name', 'gender', 'age'])
        rows = df.take(2)
        assert len(rows) == 2

    def test_select(self):
        data = [('Alice', 'F', 1), ('Bob', 'M', 2), ('Cate', 'F', 3)]
        df = spark.createDataFrame(data, ['name', 'gender', 'age'])
        rows = df.select('name', 'age').collect()
        row_1 = rows[0]
        assert row_1['name'] == 'Alice'
        assert row_1['age'] == 1
        assert 'gender' not in row_1

    def test_sort(self):
        data = [('Alice', 'F', 1), ('Bob', 'M', 2), ('Cate', 'F', 3)]
        df = spark.createDataFrame(data, ['name', 'gender', 'age'])
        rows = df.select('name', 'age').sort('age').collect()
        row_1 = rows[0]
        assert row_1['age'] == 1

    def test_function(self):
        data = [('Alice', 'F', 1), ('Bob', 'M', 2), ('Cate', 'F', 3)]
        df = spark.createDataFrame(data, ['name', 'gender', 'age'])
        rows = df.select(avg('age')).collect()
        row_1 = rows[0]
        assert row_1['avg(age)'] == 2.0

    def test_toDF(self):
        pass

    def test_udf(self):
        pass


if __name__ == '__main__':
    test = TestPySparkSQL()
    # Call a method here
    test.test_collect()
    spark.stop()