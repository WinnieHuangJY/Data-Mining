from pyspark import SparkContext
import sys
import time
from pyspark.mllib.recommendation import ALS, Rating
import random
from itertools import combinations
time_start = time.time()


sc = SparkContext('local[*]', 'Item-based CF Recommender by Winnie')
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
trb = train_business.map(lambda x: (x, 1)).collectAsMap()  # not id
tru = train_user.map(lambda x: (x, 1)).collectAsMap()
train_data = trainRDD.map(lambda x: (u_dict[x[0]], b_dict[x[1]], float(x[2])))  # (userid, businessid, rating)
# filter out the data with user or business in the test data that are new to the training data
# they will be imputed after prediction
test_data = testRDD.filter(lambda x: trb.__contains__(x[1]) and tru.__contains__(x[0])) \
    .map(lambda x: (b_dict[x[1]], u_dict[x[0]]))  # (businessid, userid)


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
# here we only use first 25 neighbors whose weight > 0
def calculate_with_neighbor(x):  # here only use neighbors whose weight > 0
        # x = (active_businessid, active_userid)

        # find businesses the active user has rated
        rated_business = user_dict[x[1]][1]  # {businessid: rating}
        active_business_users = set(business_dict[x[0]][1].keys())  # {userid} users that rated active business

        # find co-rated business
        corated_business = []  # [[businessid, {co-rated userid}]]
        for business in rated_business.keys():
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

        # calculate weights of each business
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

        # neighbors are not important if there are only few of them
        if len(corated_business) < 5:
            return ((x[1], x[0]), 0.7*user_dict[x[1]][0] + 0.3*business_dict[x[0]][0])
        num = len(corated_business)
        if num > 50:
            num = 25
        sorted_business = sorted(corated_business, key=lambda x: x[0])[:num+1]

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
added_pred = predictions.union(user_star).union(business_star).union(missing_all)  # .map(outlier)

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