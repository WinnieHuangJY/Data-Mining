from pyspark import SparkContext
import sys
import time
from pyspark.mllib.recommendation import ALS, Rating
import random
from itertools import combinations
time_start = time.time()


sc = SparkContext('local[*]', 'Item-based CF Recommender with Jaccard-based LSH by Winnie')
sc.setSystemProperty('executor.memory', '4g')
sc.setSystemProperty('drive.memory', '4g')
train_file_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]


# data preprocessing
first_row = sc.textFile(train_file_path).first()
trainRDD = sc.textFile(train_file_path).filter(lambda x: x != first_row).map(lambda x: x.split(","))
testRDD = sc.textFile(test_file_path).filter(lambda x: x != first_row).map(lambda x: x.split(","))
# dictionaries below are used to replace the raw data with index to speed up the program
train_user = trainRDD.map(lambda x: x[0]).distinct()
test_user = testRDD.map(lambda x: x[0]).distinct()
train_business = trainRDD.map(lambda x: x[1]).distinct()
test_business = testRDD.map(lambda x: x[1]).distinct()
u_dict = train_user.union(test_user).zipWithIndex().collectAsMap()
u_index = train_user.union(test_user).zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
b_dict = train_business.union(test_business).zipWithIndex().collectAsMap()
b_index = train_business.union(test_business).zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()


# first, we perform Jaccard-based LSH to find similar pairs of business
ab = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
          109, 113, 117, 119, 127, 131]
AB = random.sample(list(combinations(ab, 2)), 120)
R, M = 3, len(u_dict)
B = len(AB) // R

# perform minhash using hash functions and select the smallest (users hashed values) as the signature
def signature(x):
        signatures = []
        for a, b in AB:
            hash_values = [(a * i + b) % M for i in x[1]]
            signatures.append(min(hash_values))
        return (x[0], signatures)

# for each pair of candidates, calculate jaccard similarity in order to eliminate false positives
def jaccard(x):
        b0, b1 = set(matrix_dict[x[0]]), set(matrix_dict[x[1]])
        intersection, union = b0.intersection(b1), b0.union(b1)
        jaccard_similarity = len(intersection) / len(union)
        if jaccard_similarity >= 0.5:
            return (x[0], x[1], jaccard_similarity)
        return None

trb = train_business.map(lambda x: (x, 1)).collectAsMap()  # not id
tru = train_user.map(lambda x: (x, 1)).collectAsMap()
train_data = trainRDD.map(lambda x: (u_dict[x[0]], b_dict[x[1]], float(x[2])))  # (userid, businessid, rating)
# filter out the data with user or business in the test data that are new to the training data
# they will be imputed after prediction
test_data = testRDD.filter(lambda x: trb.__contains__(x[1]) and tru.__contains__(x[0])) \
        .map(lambda x: (b_dict[x[1]], u_dict[x[0]]))  # (businessid, userid)

# build fake characteristic matrix: each business key has value of a group of users (only '1's in the characteristic matrix)
matrix = train_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(set(x[1]))))  # (#business, [#users])
matrix_dict = matrix.collectAsMap()
# minhash + signature
minhash = matrix.map(signature)  # (#business, [signatures])
# partition the signature into bands, use band_id and band_content as the key to find candidates
# use LSH to produce candidates
bands = minhash.flatMap(lambda x: [(tuple([i] + x[1][i * R:i * R + R]), [x[0]]) for i in range(B)])\
        .reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > 1)\
        .flatMap(lambda x: list(combinations(sorted(x[1]), 2))).distinct()
similar_business = bands.map(jaccard).filter(lambda x: x is not None) \
        .map(lambda x: (x[0], x[1]))  # (business_id_1, business_id_2)
# use a dictionary to preserve the similar business of each business
similar_temp1 = similar_business.groupByKey().map(lambda x: (x[0], list(x[1])))
similar_temp2 = similar_business.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
similar_dict = similar_temp1.union(similar_temp2).collectAsMap()


# then, we perform Item-based CF, using the knowledge got from Jaccard-based LSH
# construct a dictionary for later use
def to_dict(x):
        # x = (businessid, [(userid, rating)])
        # x = (userid, [businessid, rating])
        result_dict = {}
        sum, cnt = 0, 0
        for id, rating in list(x[1]):
            result_dict[id] = rating
            sum += rating
            cnt += 1
        return (x[0], (sum / cnt, result_dict))

# construct dictionaries for later use
business_dict = train_data.map(lambda x: (x[1], (x[0], x[2]))) \
        .groupByKey().map(to_dict).collectAsMap()  # {businessid: (avg, {userid: rating})}
user_dict = train_data.map(lambda x: (x[0], (x[1], x[2]))) \
        .groupByKey().map(to_dict).collectAsMap()  # {userid: (avg, {businessid: rating})}

# using Pearson Correlation to predict the rating
# here we only use similar business to behaving as neighbor
def calculate_with_neighbor(x):
        # x = (active_businessid, active_userid)
        if similar_dict.__contains__(x[0]):
            similar = set(similar_dict[x[0]])  # [similar businessid]
        else:
            return ((x[1], x[0]), 0.7*user_dict[x[1]][0] + 0.3*business_dict[x[0]][0])
        # find businesses the active user has rated
        rated_business = set(user_dict[x[1]][1].keys()).intersection(similar)  # {businessid}
        active_business_users = set(business_dict[x[0]][1].keys())  # {userid} users that rated active business
        # find co-rated business
        corated_business = []  # [[businessid, {co-rated userid}]]
        for business in rated_business:
            rated_u = set(business_dict[business][1].keys())
            intersection = active_business_users.intersection(rated_u)
            if len(intersection) > 0:
                corated_business.append([business, intersection])
        # calculate co-rated average
        for item in corated_business:
            business, intersection = item[0], item[1]
            a_sum, b_sum = 0, 0
            for user in intersection:
                a_sum += business_dict[x[0]][1][user]
                b_sum += business_dict[business][1][user]
            item.append(a_sum / len(intersection))
            item.append(b_sum / len(intersection))
        # corated_business = [[businessid, {co-rated userid}, active_business_avg, corated_business_avg]]
        # calculate weight
        for item in corated_business:
            business, intersection, active_avg, corated_avg = item[0], item[1], item[2], item[3]
            up, d1, d2 = 0, 0, 0
            for user in intersection:
                up += (business_dict[x[0]][1][user] - active_avg) * (business_dict[business][1][user] - corated_avg)
                d1 += pow(business_dict[x[0]][1][user] - active_avg, 2)
                d2 += pow(business_dict[business][1][user] - corated_avg, 2)
            down = pow(d1, 0.5) * pow(d2, 0.5)
            if down == 0:
                item.insert(0, 0)
            else:
                item.insert(0, up / down)
        # corated_business = [[weight, businessid, {corated_user}, active_business_avg, corated_business_avg]]
        sorted_business = sorted(corated_business, key=lambda x: x[0])
        # calculate Pearson weighted sum
        molecule, denominator = 0, 0
        for item in sorted_business:
            weight, business = item[0], item[1]
            if weight <= 0:
                continue
            molecule += user_dict[x[1]][1][business] * weight
            denominator += abs(weight)
        # pay attetion to the denominator, which could be 0
        p = 0.7*user_dict[x[1]][0] + 0.3*business_dict[x[0]][0]
        if denominator != 0:
            p = molecule / denominator
        # the result might out of range
        if p < 0:
            p = 1.5
        elif p > 5:
            p = 4.5
        return ((x[1], x[0]), p)


# prediction
predictions = test_data.map(calculate_with_neighbor)


# dealing with business_id and user_id in the test data that are new to the traing data
# impute average of the user or business or whole dataset after prediction
missing_b = testRDD.filter(lambda x: (not trb.__contains__(x[1])) and (tru.__contains__(x[0]))).map(
        lambda x: (u_dict[x[0]], b_dict[x[1]]))  # missing_b (user, business)
user_used = missing_b.map(lambda x: x[0]).collect()  # (user)
user_star = predictions.filter(lambda x: x[0][0] in user_used).map(lambda x: (x[0][0], x[1])) \
        .aggregateByKey((0, 0), lambda U, v: (U[0] + v, U[1] + 1), lambda U1, U2: (U1[0] + U2[0], U1[1] + U2[1])) \
        .map(lambda x: (x[0], float(x[1][0]) / x[1][1])).join(missing_b).map(lambda x: ((x[0], x[1][1]), x[1][0]))
missing_u = testRDD.filter(lambda x: (not tru.__contains__(x[0])) and trb.__contains__(x[1])).map(
        lambda x: (b_dict[x[1]], u_dict[x[0]]))  # missing_u (business, user)
business_used = missing_u.map(lambda x: x[0]).collect()  # (business)
business_star = predictions.filter(lambda x: x[0][1] in business_used).map(lambda x: (x[0][1], x[1])) \
        .aggregateByKey((0, 0), lambda U, v: (U[0] + v, U[1] + 1), lambda U1, U2: (U1[0] + U2[0], U1[1] + U2[1])) \
        .map(lambda x: (x[0], float(x[1][0]) / x[1][1])).join(missing_u).map(lambda x: ((x[1][1], x[0]), x[1][0]))
avg = predictions.map(lambda x: x[1]).mean()
missing_all = testRDD.filter(lambda x: not tru.__contains__(x[0]) and (not trb.__contains__(x[1]))) \
        .map(lambda x: ((u_dict[x[0]], b_dict[x[1]]), avg))  # missing_all ((user, business), avg)
added_pred = predictions.union(user_star).union(business_star).union(missing_all)


# calculate RMSE to evaluate the accuracy of the system
tstdata = testRDD.map(lambda x: Rating(u_dict[x[0]], b_dict[x[1]], float(x[2])))
ratesAndPreds = tstdata.map(lambda r: ((r[0], r[1]), r[2])).join(added_pred)
MSE = pow(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean(), 0.5)
print("RMSE = " + str(MSE))


# write document
f = open(output_file_path, "w")
content = added_pred.map(lambda x: [u_index[x[0][0]], b_index[x[0][1]], x[1]]).collect()
f.write("user_id, business_id, prediction\n")
for i in content:
        f.write(i[0] + ',' + i[1] + ',' + str(i[2]) + '\n')
f.close()


# total time cost
time_end = time.time()
print("time cost:", time_end - time_start)