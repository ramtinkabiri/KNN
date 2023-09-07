import numpy as np
import math
import statistics
import matplotlib.pyplot as plt 

### distance calculator: {
def distance(a, b):
    l = len(a)
    sum_out = 0
    for c in range(0, l):
        sum_out += math.pow((a[c] - b[c]), 2)
    
    dist = math.sqrt(sum_out)
    return dist

# #check : {
# a_test = np.array([1, 3, 6, 2])
# b_test = np.array([5, 2, 0, 1])
# print(distance(a_test, b_test))
# }
### }

### k-nearest neighbor : {
def knn(train, test, k):
    q_test = np.shape(test)
    i_test = q_test[0]
    j = q_test[1]
    q_train = np.shape(train)
    i_train = q_train[0]
    t = 0
    f = 0

    for row in range(0, i_test):
        a = test[row, 0:(j-1)]
        result = test[row, -1]
        dist_list = []
        for index in range(0, i_train):
            b = train[index, 0:(j-1)]
            dist = distance(a, b)
            dist_list.append(((float(dist)), index))

        knn_list = []
        for k_neighbor in range(0, k):
            knn_list.append(min(dist_list))
            dist_list.remove(min(dist_list))

        neighbors = []
        for column in range(0, k):
            w = knn_list.pop(0)
            neighbors.append(float(w[1]))


        neighbor_result = []
        for column in range(0, k):
            e = neighbors.pop(0)
            nbresult = train[int(e), -1]
            neighbor_result.append(float(nbresult))

        r0 = neighbor_result.count(float(0))
        r1 = neighbor_result.count(float(1))
        r2 = neighbor_result.count(float(2))
        r3 = neighbor_result.count(float(3))
        vote = []
        vote.append((r0,0))
        vote.append((r1,1))
        vote.append((r2,2))
        vote.append((r3,3))
        maximum = max(vote)
        knn_result = maximum[1]
        if float(knn_result) == float(result):
            t += 1
        elif float(knn_result) != float(result):
            f += 1

    error = ((f) / (f + t))
    # print("true: ", t)
    # print("false: ", f)
    # print("error: ",error)  
    return error

### }

### read datas and make a usable matrix: {
file = open("Thyroid.txt")
data = file.readlines()
file.close()

i = len(data)
j = len(data[0].split(','))
# test_sample = 77
# print("test_sample line: ", data[test_sample].split(','))
# print("i: ", i)
# print("j: ", j)

mat = np.zeros((i, j))
for row in range(0, i):
    d = data[row].split(',')
    for column in range(0, j):
        mat[row, column] = float(d[column])

# print("test_sample row: ", mat[test_sample, :])
# # mat{ j = 7: Y ; j = [0, 6]: X }
# # mat{ j = 2,3,6 : type = binary}
### }
 
### cross validation datasets:{
#1
train1 = mat[20:100 , :]
test1 = mat[0:20 , :]
#2
train2 = np.concatenate((mat[0:20 , :], mat[40:100 , :]), axis=0)
test2 = mat[20:40 , :]
#3
train3 = np.concatenate((mat[0:40 , :], mat[60:100 , :]), axis=0)
test3 = mat[40:60 , :]
#4
train4 = np.concatenate((mat[0:60 , :], mat[80:100 , :]), axis=0)
test4 = mat[60:80 , :]
#5
train5 = mat[0:80 , :]
test5 = mat[80:100 , :]
# # check
# check_sample = test5
# print(check_sample.shape)
### }

### normalized matrix: {
a = []
for r in range(0, i):
    # با سن بهتر است به صورت عدد حقیقی برخورد کنیم ولی بدلیل خواسته سوال مبنی بر نرمالیزه کردن تمام داده ها بجز باینری ها و خروجی را اسکیل می کنیم
    for c in [0, 3, 4, 6]:  # 1,2,5:binary - 7:output 
        a.append(float(mat[r, c]))

vari = statistics.variance(a)
avg = statistics.mean(a)
# print("variance: ", vari)
# print("avrage: ", avg)

mat_normalized = np.zeros((i, j))
for row in range(0, i):
    for column in range(0, j):
        if column == 0 or column == 3 or column == 4 or column == 6:
            mat_normalized[row, column] = (abs( (mat[row, column]) - (avg) )) / (vari)
        else:
            mat_normalized[row, column] = mat[row, column]

# #check
# check_nsample = 88
# print(mat[check_nsample, :])
# print(mat_normalized[check_nsample, :])

## cross validation normalized datasets:{
#1
ntrain1 = mat_normalized[20:100 , :]
ntest1 = mat_normalized[0:20 , :]
#2
ntrain2 = np.concatenate((mat_normalized[0:20 , :], mat_normalized[40:100 , :]), axis=0)
ntest2 = mat_normalized[20:40 , :]
#3
ntrain3 = np.concatenate((mat_normalized[0:40 , :], mat_normalized[60:100 , :]), axis=0)
ntest3 = mat_normalized[40:60 , :]
#4
ntrain4 = np.concatenate((mat_normalized[0:60 , :], mat_normalized[80:100 , :]), axis=0)
ntest4 = mat_normalized[60:80 , :]
#5
ntrain5 = mat_normalized[0:80 , :]
ntest5 = mat_normalized[80:100 , :]
## }

### }

### outputs: {
avrage = []
for k in range(1,11):
    out1 = knn(train= train1, test= test1, k= k)
    #print("1-%i: "%k, out1)
    out2 = knn(train= train2, test= test2, k= k)
    #print("2-%i: "%k, out2)
    out3 = knn(train= train3, test= test3, k= k)
    #print("3-%i: "%k, out3)
    out4 = knn(train= train4, test= test4, k= k)
    #print("4-%i: "%k, out4)
    out5 = knn(train= train5, test= test5, k= k)
    #print("5-%i: "%k, out5)
    avg_out = (out1 + out2 + out3 + out4 + out5) / 5
    print("avg_out%i :"%k , avg_out)
    avrage.append(avg_out)

### }
print("---------------------------")
### normalized outputs: {
navrage = []
for k in range(1,11):
    nout1 = knn(train= ntrain1, test= ntest1, k= k)
    #print("1-%i: "%k, out1)
    nout2 = knn(train= ntrain2, test= ntest2, k= k)
    #print("2-%i: "%k, out2)
    nout3 = knn(train= ntrain3, test= ntest3, k= k)
    #print("3-%i: "%k, out3)
    nout4 = knn(train= ntrain4, test= ntest4, k= k)
    #print("4-%i: "%k, out4)
    nout5 = knn(train= ntrain5, test= ntest5, k= k)
    #print("5-%i: "%k, out5)
    navg_out = (nout1 + nout2 + nout3 + nout4 + nout5) / 5
    print("navg_out%i :"%k , navg_out)
    navrage.append(navg_out)

### }

### error avrage diagram : {
fig, ax = plt.subplots()
k_list= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ax.plot(k_list, avrage, 'b', linewidth=2.0)
ax.plot(k_list, navrage, 'g', linewidth=2.0)
plt.show()

### }