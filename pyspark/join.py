from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()
sc = spark.sparkContext
rdd1 = sc.parallelize([("m",55),("e",57),("e",58),("s",54)])
rdd2 = sc.parallelize([("m",60), ("m", 70),("s",61),("h",63)])
# inner join
result = rdd1.join(rdd2) # Join is inner join
print(result.collect())

# ful outer Join
result = rdd1.fullOuterJoin(rdd2)
print("Full outer join ")
print(result.collect())

# cartesion join
result = rdd1.cartesian(rdd2)
print("All combination of rdd1 joined with rdd 2")
print(result.collect())