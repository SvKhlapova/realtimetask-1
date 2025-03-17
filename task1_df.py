from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, log, split, udf, regexp_replace, lower, explode, count, concat_ws, broadcast
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder \
    .appName("Bigrams") \
    .config("spark.sql.shuffle.partitions", "500") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.catalog.clearCache()

stop_words_file = "/data/wiki/stop_words_en-xpo6.txt"
dataset = "/data/wiki/en_articles_part"
pair_thresh = 500

stop_words = spark.read.text(stop_words_file).withColumnRenamed("value", "stop_word")
stop_words_list = stop_words.rdd.flatMap(lambda x: x).collect()
stop_words_list = [word.strip().lower() for word in stop_words_list]
stop_words_list = spark.sparkContext.broadcast(stop_words_list)

source_text = spark.read.text(dataset).withColumnRenamed("value", "text").cache()

articles = source_text.withColumn("words", split(lower(col("text")), "\s+")) \
    .withColumn("words", expr("transform(words, x -> regexp_replace(x, '[^0-9a-z]', ''))")) \
    .cache()

@udf(ArrayType(StringType()))
def remove_stop_words(words):
    return [word for word in words if word not in stop_words_list.value]

filtered_words = articles.withColumn("filtered_words", remove_stop_words(col("words")))

words = filtered_words.withColumn("word", explode(col("filtered_words"))) \
    .select("word")

word_counts = words.groupBy("word").agg(count("word").alias("word_count")).cache()

total_words = words.count()

@udf(ArrayType(StringType()))
def create_bigrams(words):
    return ["{}_{}".format(words[i], words[i+1]) for i in range(len(words)-1)]

bigrams = filtered_words.withColumn("bigrams", create_bigrams(col("filtered_words"))) \
    .withColumn("bigram", explode(col("bigrams"))) \
    .withColumn("word1", split(col("bigram"), '_')[0]) \
    .withColumn("word2", split(col("bigram"), '_')[1]) \
    .filter(~col("word1").isin(stop_words_list.value) & ~col("word2").isin(stop_words_list.value)) \
    .select("bigram", "word1", "word2")

total_bigrams = bigrams.count()

bigram_counts = bigrams.groupBy("bigram", "word1", "word2").agg(count("bigram").alias("bigram_count")).cache()

word_counts_word1 = word_counts.withColumnRenamed("word", "word1").withColumnRenamed("word_count", "word1_count")
word_counts_word2 = word_counts.withColumnRenamed("word", "word2").withColumnRenamed("word_count", "word2_count")

bigram_counts = bigram_counts.join(
    broadcast(word_counts_word1),
    bigram_counts["word1"] == word_counts_word1["word1"],
    "inner"
)

bigram_counts = bigram_counts.join(
    broadcast(word_counts_word2),
    bigram_counts["word2"] == word_counts_word2["word2"],
    "inner"
)

bigram_counts = bigram_counts.withColumn(
    "pmi", log((col("bigram_count") / total_bigrams) / ((col("word1_count") / total_words) * (col("word2_count") / total_words)))
)

bigram_counts = bigram_counts.withColumn("npmi", -col("pmi") / log(col("bigram_count") / total_bigrams))

result = bigram_counts.filter(col("bigram_count") > pair_thresh) \
                      .orderBy(col("npmi").desc()) \
                      .select("bigram", "npmi") \
                      .limit(39)

for pair in result.collect():
    print(pair["bigram"])

spark.stop()