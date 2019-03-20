from pyspark import SparkContext
import sys
import time
time_start = time.time()
from itertools import combinations
import random


sc = SparkContext('local[*]', 'Winnie_JaccardLSH')
sc.setSystemProperty('executor.memory', '4g')
sc.setSystemProperty('drive.memory', '4g')
sc.setSystemProperty('spark.driver.maxResultSize', '10g')
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
ab = [2, 3, 5, 7, 11,   13, 17, 19, 23, 29,   31, 37, 41, 43, 47,   53, 59, 61, 67, 71,   73, 79, 83, 89, 97,   101, 103, 107, 109, 113, 117, 119, 127, 131]
AB = random.sample(list(combinations(ab, 2)), 120)
R = 3
B = len(AB)//R


# used for preprocessing data
def pre(x):
    row = x.split(",")
    return (row[1], row[0])


# perform minhash using hash functions and select the smallest (users hashed values) as the signature
def signature(x):
    signatures = []
    for a,b in AB:
        hash_values = [(a*i+b)%M for i in x[1]]
        signatures.append(min(hash_values))
    return (x[0], signatures)


# partition the signature into bands, use band_id and band_content as the key to find candidates
def banding(x):
    band = []
    for i in range(B):
        band.append((tuple([i]+x[1][i*R:i*R+R]), [x[0]]))
    return band


first_row = sc.textFile(input_file_path).first()
textRDD = sc.textFile(input_file_path, B).filter(lambda x: x != first_row).map(pre)  # (business, user)
u_dict = textRDD.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
b_temp = textRDD.map(lambda x: x[0]).distinct().zipWithIndex()
b_dict = b_temp.collectAsMap()
b_index = b_temp.map(lambda x: (x[1],x[0])).collectAsMap()
M = len(u_dict)

# build fake characteristic matrix: each business key has value of a group of users (only '1's in the characteristic matrix)
matrix = textRDD.map(lambda x: (b_dict[x[0]], u_dict[x[1]])).groupByKey().map(lambda x: (x[0], list(set(x[1]))))  # (#business, [#users])
matrix_dict = matrix.collectAsMap()

# minhash + signature
minhash = matrix.map(signature)  # (#business, [signatures])

# use LSH to produce candidates
bands = minhash.flatMap(lambda x: [(tuple([i]+x[1][i*R:i*R+R]), [x[0]]) for i in range(B)]).reduceByKey(lambda a,b: a+b) \
    .filter(lambda x: len(x[1])>1).flatMap(lambda x: list(combinations(sorted(x[1]), 2))).distinct()


# for each pair of candidates, calculate jaccard similarity in order to eliminate false positives
def jaccard(x):
    b0, b1 = set(matrix_dict[x[0]]), set(matrix_dict[x[1]])
    intersection, union = b0.intersection(b1), b0.union(b1)
    jaccard_similarity = len(intersection)/len(union)
    if jaccard_similarity >= 0.5:
        return (x[0], x[1], jaccard_similarity)
    return None


# get the result
results = bands.map(jaccard).filter(lambda x: x is not None)\
    .map(lambda x: (tuple(sorted([b_index[x[0]], b_index[x[1]]])), x[2])).sortByKey().collect()

# write document
f = open(output_file_path, "w")
f.write("business_id_1, business_id_2, similarity\n")
for row in results:
    f.write(row[0][0]+','+row[0][1]+','+str(row[1])+'\n')

time_end = time.time()
print("time cost:", time_end-time_start)