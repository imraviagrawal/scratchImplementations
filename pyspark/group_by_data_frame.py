from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()

"""
Find top 2 tracks per genre 
"""
songs = ["as it was", "I aint worried", "I like you", "I aint worried", "as it was", "I like you"]
genre = ["pop", "rock", "pop", "rock", "pop", "rock"]
platform = ["spotify", "spotify", "youtube", "billboard", "billboard", "billboard"]
times = [10, 15, 5, 1, 4, 6]
df = spark.createDataFrame(zip(songs,genre, times, platform),['songs','genre', "times", "platform"])
rdd=df.rdd

map = rdd.map(lambda x: ((x.genre,x.songs), (x.times, x.platform))) #((genre, song): (times, platform)) # reduce by number
count_genre_song = map.reduceByKey(lambda x, y: (x[0]+y[0], "_".join([x[1],y[1]]))) #(genre, song): (times, platform_12)

# song and genre
genreMap = count_genre_song.map(lambda x: ((x[0][0]), (x[1][0], x[0][1], x[1][1]))) #(genre):(song, times, platform_12)
reduce = genreMap.reduceByKey(lambda x, y: (x, y)) # (genre, [(song, times, platform_12)....] list of all songs inside it

# sort for each key
reduce_flat = reduce.flatMapValues(lambda x: sorted(x, key=lambda a: a[1][0], reverse=False)[:2])  # (genre, [(song, times, platform_12)....] list of all songs inside it
print(reduce_flat.collect())

# sort with filter
reduce_flat_filter = reduce.flatMapValues(lambda x: sorted(x, key=lambda a: a[1][0], reverse=False)[:2])
reduce_flat_filter=reduce_flat_filter.filter(lambda x:x[1][0]>10) # filter if the vaues are less the 10
print(reduce_flat_filter.collect())