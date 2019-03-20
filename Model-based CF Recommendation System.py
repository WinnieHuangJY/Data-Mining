from pyspark import SparkContext
import sys
import time
from pyspark.mllib.recommendation import ALS, Rating
import random
from itertools import combinations
time_start = time.time()


sc = SparkContext('local[*]', 'Model-based CF Recommender by Winnie')
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
train_data = trainRDD.map(lambda x: Rating(u_dict[x[0]], b_dict[x[1]], float(x[2])))
test_data = testRDD.map(lambda x: Rating(u_dict[x[0]], b_dict[x[1]], float(x[2])))

# build model
rank = 2
numIterations = 21
model = ALS.train(train_data, rank, numIterations, 0.2)

# perform the model on training data
testdata = test_data.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

# there are some business_id or user_id or both new to the model,
# impute these missing data with averages regarding to user, business or the whole dataset
trb = train_business.map(lambda x: (x, 1)).collectAsMap()
missing_b = testRDD.filter(lambda x: not trb.__contains__(x[1])).map(lambda x: (u_dict[x[0]], b_dict[x[1]]))  # missing_b (user, business)
user_used = missing_b.map(lambda x: x[0]).collect()  # (user)
user_star = predictions.filter(lambda x: x[0][0] in user_used).map(lambda x: (x[0][0], x[1])) \
        .aggregateByKey((0, 0), lambda U, v: (U[0] + v, U[1] + 1), lambda U1, U2: (U1[0] + U2[0], U1[1] + U2[1])) \
        .map(lambda x: (x[0], float(x[1][0]) / x[1][1])).join(missing_b).map(lambda x: ((x[0], x[1][1]), x[1][0]))
tru = train_user.map(lambda x: (x, 1)).collectAsMap()
missing_u = testRDD.filter(lambda x: not tru.__contains__(x[0])).map(lambda x: (b_dict[x[1]], u_dict[x[0]]))  # missing_u (business, user)
business_used = missing_u.map(lambda x: x[0]).collect()  # (business)
business_star = predictions.filter(lambda x: x[0][1] in business_used).map(lambda x: (x[0][1], x[1])) \
    .aggregateByKey((0, 0), lambda U, v: (U[0] + v, U[1] + 1), lambda U1, U2: (U1[0] + U2[0], U1[1] + U2[1])) \
    .map(lambda x: (x[0], float(x[1][0]) / x[1][1])).join(missing_u).map(lambda x: ((x[1][1], x[0]), x[1][0]))
avg = predictions.map(lambda x: x[1]).mean()
missing_all = testRDD.filter(lambda x: not tru.__contains__(x[0]) and (not trb.__contains__(x[1]))) \
    .map(lambda x: ((u_dict[x[0]], b_dict[x[1]]), avg))  # missing_all ((user, business), avg)
added_pred = predictions.union(user_star).union(business_star).union(missing_all)

# calculate RMSE to evaluate the accuracy
ratesAndPreds = test_data.map(lambda r: ((r[0], r[1]), r[2])).join(added_pred)
MSE = pow(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean(), 0.5)
print("RMSE = " + str(MSE))

#write document
f = open(output_file_path, "w")
content = added_pred.map(lambda x: [u_index[x[0][0]], b_index[x[0][1]], x[1]]).collect()
f.write("user_id, business_id, prediction\n")
for i in content:
    f.write(i[0]+','+i[1]+','+str(i[2])+'\n')
f.close()

# total time coast of the program
time_end = time.time()
print("time cost:", time_end - time_start)
