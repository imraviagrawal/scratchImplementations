from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()
sc = spark.sparkContext
rdd1 = sc.parallelize([("m",55),("e",57),("e",58),("s",54)])
rdd2 = sc.parallelize([("m",60), ("m", 70),("s",61),("h",63)])
leftjoinrdd = rdd1.leftOuterJoin(rdd2)
print(leftjoinrdd.collect())