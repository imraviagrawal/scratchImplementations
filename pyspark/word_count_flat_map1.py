from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()

"""
Iterate through the list of data, generate count and filter the results
"""

data = ["Project Gutenberg’s Alice’s Adventures",
"in","Wonderland Project Gutenberg’s Adventures",
"in","Wonderland Project Gutenberg’s"]

rdd=spark.sparkContext.parallelize(data)

flatmap = rdd.flatMap(lambda x: x.split(" ")) # "sentence": ["word1", "word2", 'word3']
# mapper fuction
map = flatmap.map(lambda x: (x, 1)) # <key>:<key, value>

# sort and shuffle
reduce = map.reduceByKey(lambda x, y: (x+y)) # <value, value> --> <value+value>
filter = reduce.filter(lambda x: x[1]>2) # <key, value>: <key, value>
result = filter.collect()
print(result)