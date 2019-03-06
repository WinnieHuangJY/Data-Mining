from pyspark import SparkContext
import os
import psutil
import sys
import json
import time
time_start = time.time()
from itertools import combinations
from collections import Counter
from operator import add

# os.environ['PYSPARK_PYTHON']="/usr/local/bin/python3.6"
# os.environ['PYSPARK_DRIVER_PYTHON']="/usr/local/bin/python3.6"
sc = SparkContext('local[*]', 'JiayingHW2')
# k_thred = 70
# original_support = 50
# input_file_path = 'D:/workspace/PycharmProjects/INF553_hw1/data/task2_data.csv'
# output_file_path = "D:/workspace/PycharmProjects/INF553_hw1/data/hw2_2.txt"
k_thred = int(sys.argv[1])
original_support = int(sys.argv[2])
support = original_support//2
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]


def AP(x):
    # 找出frequent single
    baskets = []
    count = [{}, {}]
    frequents = [(1, []), (2, [])]
    for tupl in x:
        # print(tupl)
        baskets.append(tupl[1])
        for key in tupl[1]:
            count[0][key] = count[0].get(key, 0) + 1
    for i in sorted(count[0].keys()):
        if count[0][i] >= support:
            frequents[0][1].append(i)
    print("frequent single:", len(frequents[0][1]))
    # 找出frequent pair
    for basket in baskets:
        basket = [i for i in basket if i in frequents[0][1]]
        # basket = list(set(basket).intersection(set(frequents[0][1])))
        pairs = combinations(basket, 2)
        for pair in pairs:
            pair = tuple(sorted(pair))
            count[1][pair] = count[1].get(pair, 0) + 1
    for i in sorted(count[1].keys()):
        if count[1][i] >= support:
            frequents[1][1].append(i)
    print("frequent pair:", len(frequents[1][1]))
    # print("frequent single & pairs:", frequents)
    # print('\n')
    # 循环找其他
    c = 3
    maxlen = len(frequents[0][1])
    # 建立可能需要被计数的项，看实际basket中是否存在
    while c <= maxlen:
        # print(c)
        count.append({})
        #暴力找需要被计数的项
        candidates = combinations(frequents[0][1], c)
        for candidate in candidates:
            if set(combinations(candidate, c-1)).issubset(frequents[c-2][1]):
                count[c - 1][candidate] = 0
        #根据被计数的项数basket
        for candidate in count[c-1]:
            for basket in baskets:
                if set(candidate).issubset(basket):
                    count[c - 1][candidate] += 1
        frequents.append((c, [i for i in count[c-1] if count[c-1][i] >= support]))
        print("frequent "+str(c)+": ", len(frequents[c-1][1]))
        if not frequents[c-1][1]:
            frequents.pop()
            break
        c += 1
        if len(frequents[c-2][1]) < c:
            break
    lp = 0
    for i in frequents:
        lp += len(i[1])
    print("frequent itemset:", lp)
    # print('\n\n\n')
    media = []
    for i in frequents:
        media = media + i[1]
    return media


def AP_local(x):
    # 找出frequent single
    baskets = []
    count = [{}, {}]
    frequents = [(1, []), (2, [])]
    for tupl in x:
        # print(tupl)
        bk = list(set(tupl[1]))
        baskets.append(sorted(bk))
        # baskets.append(sorted(tupl[1]))
        for key in sorted(tupl[1]):
            count[0][key] = count[0].get(key, 0) + 1
    for i in count[0].keys():
        if count[0][i] >= support:
            frequents[0][1].append(i)
    print("frequent single:", len(frequents[0][1]))
    # 找出frequent pair
    for i in range(len(baskets)):
        single = [x for x in baskets[i] if x in frequents[0][1]]
        # print(single)
        cb = combinations(baskets[i], 2)
        baskets[i] = [single, []]
        for key in cb:
            # key = tuple(sorted(key))
            count[1][key] = count[1].get(key, 0) + 1
            baskets[i][1].append(key)
    for i in sorted(count[1].keys()):
        if count[1][i] >= support:
            frequents[1][1].append(i)
    print("frequent pair:", len(frequents[1][1]))
    # print("frequent pair:", frequents[1][1])
    #找frequent x
    x = 3
    max_len = len(frequents[0][1])
    while x <= max_len:
        count.append({})
        for j in range(len(baskets)):
            # filter frequent x-1
            baskets[j][1] = [i for i in baskets[j][1] if i in frequents[x-2][1]]
            # # update frequent single
            temp = []
            # update_single = []
            for i in baskets[j][1]:
                for k in i:
                    temp.append(k)
            baskets[j][0] = [i for i in baskets[j][0] if i in temp]
            # 构造 x
            baskets[j][1] = [tuple(sorted(i)) for i in combinations(baskets[j][0], x) if set(combinations(i, x-1)).issubset(baskets[j][1])]
            # 数 x
            for key in baskets[j][1]:
                count[x-1][key] = count[x-1].get(key, 0) + 1
        frequents.append((x, [i for i in count[x-1].keys() if count[x-1][i] >= support]))
        print("frequent " + str(x) + ": ", len(frequents[x - 1][1]))
        if not frequents[x-1][1]:
            frequents.pop()
            break
        x += 1
        if len(frequents[x-2][1]) < x:
            break
    lp = 0
    for i in frequents:
        lp += len(i[1])
    print("frequent itemset:", lp)
    # print('\n\n\n')
    media = []
    for i in frequents:
        media = media + i[1]
    return media


def AP_dict(x):
    # 找出frequent single
    baskets = []
    count = {}
    # frequents = [(1, []), (2, [])]
    frequents = {}
    for tupl in x:
        bk = list(set(tupl[1]))
        baskets.append(sorted(bk))
        # baskets.append(sorted(tupl[1]))
        for key in sorted(tupl[1]):
            count[key] = count.get(key, 0) + 1
    for i in count.keys():
        if count[i] >= support:
            frequents[i] = 1
    # print("frequent single:", len(frequents))
    max_len = len(frequents)
    # 找出frequent pair
    for i in range(len(baskets)):
        single = [x for x in baskets[i] if frequents.__contains__(x)]
        cb = combinations(baskets[i], 2)
        baskets[i] = [single, {}, True]
        for key in cb:
            # key = tuple(sorted(key))
            count[key] = count.get(key, 0) + 1
            baskets[i][1][key] = 1
    for i in count.keys():
        if count[i] >= support:
            frequents[i] = 1
    # print("frequent pair + single:", len(frequents))
    #找frequent x
    x = 3
    while x <= max_len:
        flen = len(frequents)
        for j in range(len(baskets)):
            if baskets[j][2] == False:
                continue
            # filter frequent x-1
            flt = {}
            update_single = []
            for i in baskets[j][1]:
                if frequents.__contains__(i):
                    flt[i] = 1
                    for k in i:
                        # what.add(k)
                        update_single.append(k)
            baskets[j][0] = [p for p in baskets[j][0] if p in update_single]
            # filter之后若frequent x-1数量少于x，则这一basket不用继续了
            if len(flt) < x:
                baskets[j][2] = False
                continue
            # filter frequent single
            # baskets[j][0] = [k for k in baskets[j][0] if flt.__contains__(k)]
            # 构造 x
            # baskets[j][1] = [i for i in combinations(baskets[j][0], x) if set(combinations(i, x-1)).issubset(baskets[j][1])]
            baskets[j][1] = {}
            for i in combinations(baskets[j][0], x):
                check = True
                for item in combinations(i, x-1):
                    if not flt.__contains__(item):
                        check = False
                        break
                if check:
                    baskets[j][1][i] = 1
            # 数 x
            for key in baskets[j][1]:
                count[key] = count.get(key, 0) + 1
        for i in count.keys():
            if count[i] >= support:
                frequents[i] = 1
        # 如果过滤之后frequent x为空，删掉这一层并跳出
        if len(frequents) == flen:
            # count.pop()
            break
        #如果frequent x的数量小于x+1, 则不用继续
        x += 1
        if len(frequents)-flen < x:
            break
    return frequents.keys()


def phase2(x):
    count = []
    cnt = {}
    baskets = []
    # maxlen = len(CDD[-1])
    for tupl in x:
        baskets.append(tupl[1])
    #     for candidate in CDD:
    #         if type(candidate) == type('sss'):
    #             if candidate in tupl[1]:
    #                 cnt[candidate] = cnt.get(candidate, 0) + 1
    #         else:
    #             if set(candidate).issubset(tupl[1]):
    #                 cnt[candidate] = cnt.get(candidate, 0) + 1
    # for i in cnt:
    #     count.append((i, cnt[i]))

    for candidate in CDD:
        num = 0
        for basket in baskets:
            if type(candidate) == type('sss'):
                if candidate in basket:
                    num += 1
            else:
                if set(candidate).issubset(basket):
                    num += 1
        count.append((candidate, num))
    return count


textRDD = sc.textFile(input_file_path).coalesce(2).filter(lambda x: x != "user_id,business_id")\
    .map(lambda x: tuple(x.split(',')))\
    .groupByKey() \
    .filter(lambda x: len(set(x[1])) > k_thred)
CDD = textRDD.mapPartitions(AP_dict).distinct().collect()
# print(len(CDD))
time1 = time.time()
# print("part1 time: ", time1 - time_start)
PH2 = textRDD.mapPartitions(phase2)\
        .reduceByKey(add) \
        .filter(lambda x: x[1] >= original_support) \
        .map(lambda x:  x[0]).collect()
        # .sortByKey(False) \
        # .take(100)
# print(PH2)
# print(len(PH2))
out = {}
for i in PH2:
    if type(i) == type("sss"):
        if out.__contains__(1):
            out[1].append(i)
        else:
            out[1] = [i]
    else:
        key = len(i)
        if out.__contains__(key):
            out[key].append(i)
        else:
            out[key] = [i]
# print(out)
fo = open(output_file_path, "w", encoding="utf-8")
fo.write("Candidates:")
cand = {1:[]}
for i in CDD:
    if type(i) == type("sss"):
        cand[1].append(i)
    else:
        key = len(i)
        if cand.__contains__(key):
            cand[key].append(i)
        else:
            cand[key] = [i]
for i in sorted((cand.keys())):
    if i == 1:
        fo.write("\n")
        temp = sorted(cand[i])
        for j in range(len(cand[i])):
            if j == 0:
                fo.write("('" + str(temp[j]) + "')")
            else:
                fo.write(",('" + str(temp[j]) + "')")
    else:
        fo.write("\n\n")
        temp = sorted(cand[i])
        for j in range(len(cand[i])):
            if j == 0:
                fo.write(str(temp[j]))
            else:
                fo.write("," + str(temp[j]))
    # fo.write(str(sorted(cand[i]))[1:-1])
fo.write("\n\nFrequent Itemsets:\n")
for i in sorted(out.keys()):
    # print(i, len(out[i]))
    # print(str(i)+":", len(out[i]))
    # print("\n")
    if i == 1:
        temp = sorted(out[i])
        for j in range(len(out[i])):
            if j == 0:
                fo.write("('" + str(temp[j]) + "')")
            else:
                fo.write(",('" + str(temp[j]) + "')")
    else:
        fo.write("\n\n")
        temp = sorted(out[i])
        for j in range(len(out[i])):
            if j == 0:
                fo.write(str(temp[j]))
            else:
                fo.write("," + str(temp[j]))
    # fo.write(str(sorted(out[i]))[1:-1])
# print(len(PH2))
fo.close()
time_end = time.time()
print("Duration:", time_end - time_start)