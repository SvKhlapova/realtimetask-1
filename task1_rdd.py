from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re
import numpy as np
import math

config = SparkConf().setAppName("collocation").setMaster("yarn")
sc = SparkContext(conf=config)

dataset = "/data/wiki/en_articles_part"
bigrams_thresh = 500

stop_words_file = "/data/wiki/stop_words_en-xpo6.txt"
stop_words = sc.textFile(stop_words_file).collect()

stop_words_bcast = sc.broadcast(stop_words)

def parse_article(line):
    try:
        article_id, text = line.rstrip().split('\t', 1)
        text = re.sub("^\W+|\W+$", "", text, flags=re.UNICODE)
        words = re.split("\W*\s+\W*", text, flags=re.UNICODE)
        return words
    except ValueError as e:
        return []

def lower(words):
    return [word.lower() for word in words]

def filter_stop(words):
    return [word for word in words if word not in stop_words_bcast.value]

def bigram(words):
    out = []
    for w1, w2 in zip(words, words[1:]):
        out.append((w1.lower() + "_" + w2.lower(), 1))
    return out

source_text = (sc.textFile(dataset, 16)
         .map(parse_article)
         .map(lower)
         .map(filter_stop)
        ).cache()

words = (source_text.flatMap(lambda wds : [(word, 1) for word in wds])
         .reduceByKey(lambda x,y: x+y)
        ).cache()

words_total = words.map(lambda value: value[1]).sum()
words_total = sc.broadcast(words_total)

words_count_map = words.collectAsMap()
words_count_map = sc.broadcast(words_count_map)

bigram = (source_text.flatMap(bigram)
         .reduceByKey(lambda x,y : x+y)
        ).cache()

bigram_total = bigram.map(lambda value: value[1]).sum()
bigram_total = sc.broadcast(bigram_total)

bigrams_counts = bigram.countByValue()

def npmi(value):
    bigrams, count = value
    w1, w2 = bigrams.split("_")
    w1_count = words_count_map.value[w1]
    w2_count = words_count_map.value[w2]

    bigrams_prob = float(count) / bigram_total.value
    w1_prob = float(w1_count) / words_total.value
    w2_prob = float(w2_count) / words_total.value

    pmi = math.log(bigrams_prob / (w1_prob * w2_prob))
    npmi = pmi / (-1 * math.log(bigrams_prob))
    return (bigrams, npmi)

npmi = (bigram
        .filter(lambda value: value[1] > bigrams_thresh)
        .map(lambda value: npmi(value))
        .sortBy(lambda value: value[1], ascending=False)
       ).cache()

for bigrams, value in npmi.take(39):
    print(bigrams)