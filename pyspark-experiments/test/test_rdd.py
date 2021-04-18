from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('test_rdd').getOrCreate()
sc = spark.sparkContext


class TestRDD():
    # Creations
    def test_create_from_dataframe(self):
        df = spark.range(10).toDF('id')
        rdd = df.rdd
        rows = rdd.collect()
        assert len(rows) == 10
        assert rows[9]['id'] == 9

    def test_create_from_collection(self):
        data = [1, 2, 3, 4]
        rdd = sc.parallelize(data, 2)
        list_1 = rdd.collect()
        assert list_1 == [1, 2, 3, 4]
        list_2 = rdd.glom().collect()
        assert list_2 == [[1, 2], [3, 4]]

    def test_create_from_file(self):
        pass

    # Transformations
    def test_map(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        rdd_1 = rdd.map(lambda word: (word, word[0], len(word)))
        list_1 = rdd_1.collect()
        assert list_1[0] == ('The', 'T', 3)

    def test_filter(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        rdd_1 = rdd.map(lambda word: (word, word[0], len(word)))
        rdd_2 = rdd_1.filter(lambda record: record[2] == 5)
        list_2 = rdd_2.collect()
        assert list_2 == [('quick', 'q', 5), ('brown', 'b', 5), ('jumps', 'j', 5)]

    def test_sortBy(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        rdd_1 = rdd.sortBy(lambda word: len(word))
        list_1 = rdd_1.take(5)
        assert list_1 == ['The', 'fox', 'the', 'dog', 'over']

    # Partition Transformations
    def test_mapPartitions(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        rdd_1 = rdd.mapPartitions(lambda part: [word[::-1] for word in part])
        list_1 = rdd_1.collect()
        assert list_1 == ['ehT', 'kciuq', 'nworb', 'xof', 'spmuj', 'revo', 'eht', 'yzal', 'god']

    def test_foreachPartition(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)

        def func(partition):
            for word in partition:
                print(word[::-1])

        rdd.foreachPartition(func)

    # Actions
    def test_count(self):
        data = range(1, 5)
        rdd = sc.parallelize(data)
        cnt = rdd.count()
        assert cnt == 4

    def test_reduce(self):
        data = range(1, 5)
        rdd = sc.parallelize(data)
        product = rdd.reduce(lambda x, y: x * y)
        assert product == 24

    # Pair RDDs
    def test_keyBy_and_mapValues(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        pair_rdd = sc.parallelize(words).keyBy(lambda word: word.lower()[0])
        rdd_1 = pair_rdd.mapValues(lambda word: word.upper())
        list_1 = rdd_1.take(3)
        assert list_1 == [('t', 'THE'), ('q', 'QUICK'), ('b', 'BROWN')]
        list_2 = rdd_1.keys().collect()
        assert list_2 == ['t', 'q', 'b', 'f', 'j', 'o', 't', 'l', 'd']
        list_3 = rdd_1.values().collect()
        assert list_3[0] == 'THE'

    def test_countByKey(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        pair_rdd = sc.parallelize(words).map(lambda word: (word.lower()[0], word.upper()))
        d = pair_rdd.countByKey()
        assert list(d.items()) == [('t', 2), ('q', 1), ('b', 1), ('f', 1), ('j', 1), ('o', 1), ('l', 1), ('d', 1)]

    def test_reduceByKey(self):
        pair_rdd = sc.parallelize([('a', 1), ('b', 2), ('c', 3), ('b', 2), ('a', 1)])
        rdd_1 = pair_rdd.reduceByKey(lambda x, y: x*y)
        list_1 = rdd_1.collect()
        assert list_1 == [('a', 1), ('b', 4), ('c', 3)]

    # Broadcast Variable
    def test_BV(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        bv_data = {'e': 5, 'j': 10, 'o': 15, 't': 20, 'y': 25}
        bv = sc.broadcast(bv_data)
        bv_value = bv.value
        rdd_1 = rdd.map(lambda word: bv_value.get(word.lower()[0], -1))
        list_1 = rdd_1.collect()
        assert list_1 == [20, -1, -1, -1, 10, 15, 20, -1, -1]

    # Accumulator
    def test_accumulator(self):
        words = 'The quick brown fox jumps over the lazy dog'.split(' ')
        rdd = sc.parallelize(words, 2)
        first_acc = sc.accumulator(value=0)

        def func(word):
            if len(word) == 3:
                first_acc.add(1)

        rdd.foreach(func)
        assert first_acc.value == 4


if __name__ == '__main__':
    test = TestRDD()
    # Call a method here
    test.test_accumulator()
    spark.stop()