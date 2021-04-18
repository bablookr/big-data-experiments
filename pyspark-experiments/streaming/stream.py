from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from time import sleep

spark = SparkSession.builder.appName('streaming').getOrCreate()
sc = spark.sparkContext
ssc = StreamingContext(sc, 1)
ssc.checkpoint('/tmp')

lines = ssc.socketTextStream('0.0.0.0', 301)
words = lines.flatMap(lambda s: s.split(' '))
pairs = words.map(lambda word: (word, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)

counts.pprint()

ssc.start()
sleep(5)
ssc.stop(stopSparkContext=False, stopGraceFully=True)