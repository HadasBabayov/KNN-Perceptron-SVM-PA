# Hadas Babayov 322807629
import sys
import numpy as np

# Normalize the values according to the MIN-MAX normalization.
def minmax(list, train):
    normal = {}
    for i in range(5):
        normal[i] = []

    for arr in train:
        normal[0].append(arr[0])
        normal[1].append(arr[1])
        normal[2].append(arr[2])
        normal[3].append(arr[3])
        normal[4].append(arr[4])

    for index, arr in enumerate(list):
        for i in range(len(arr)):
            arr[i] = (arr[i] - np.amin(normal[i])) / (np.amax(normal[i]) - np.amin(normal[i]))

# Normalize the values according to the Z-SCORE normalization.
def zscore(list, train):
    normal = {}
    for i in range(5):
        normal[i] = []

    for arr in train:
        normal[0].append(arr[0])
        normal[1].append(arr[1])
        normal[2].append(arr[2])
        normal[3].append(arr[3])
        normal[4].append(arr[4])

    for index, arr in enumerate(list):
        for i in range(len(arr)):
            arr[i] = (arr[i] - np.mean(normal[i])) / (np.std(normal[i]))

# Check which number appears most in array.
def max_show_in_array(arr):
    num_of_show = [0, 0, 0]
    for i in range(len(arr)):
        if arr[i] == 0:
            num_of_show[0] += 1
        if arr[i] == 1:
            num_of_show[1] += 1
        if arr[i] == 2:
            num_of_show[2] += 1

    max_show = 0
    max_val = 0
    for i, num in enumerate(num_of_show):
        if num > max_show:
            max_show = num
            max_val = i
    return max_val

# KNN algorithm - return the predicted value that calculate by k closest examples in the train.
def KNN(train_x_data, tarin_y_data, test_x, k):
    test_y = []
    for x in test_x:
        allDist = []
        k_nearest = []
        for i, tr_x in enumerate(train_x_data):
            d = np.linalg.norm(x - tr_x)
            allDist.append((i, d))
        allDist.sort(key=lambda tuple: tuple[1])
        for j in range(k):
            k_nearest.append(tarin_y_data[allDist[j][0]])
        test_y.append(max_show_in_array(k_nearest))
    return test_y

# Train perceptron algorithm - return weight vectors.
def train_perceptron(train_x_data, train_y_data):
    w = np.zeros((3, 6))
    train_x_data = np.insert(train_x_data, 0, np.ones(train_x_data.shape[0]), axis=1)
    for i in range(22):
        for x_i, y_i in zip(train_x_data, train_y_data):
            #rate = 1 / (i + 1)
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != y_i:
                w[int(y_i)] += 0.15 * x_i
                w[y_hat] -= 0.15 * x_i
    return w

# Train PA algorithm - return weight vectors.
def train_pa(train_x_data, train_y_data):
    w = np.zeros((3, 6))
    train_x_data = np.insert(train_x_data, 0, np.ones(train_x_data.shape[0]), axis=1)
    for i in range(6):
        for x_i, y_i in zip(train_x_data, train_y_data):
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != y_i:
                tao = max(0, 1 - np.dot(w[int(y_i)], x_i) + np.dot(w[int(y_hat)], x_i)) / (
                            2 * pow(np.linalg.norm(x_i), 2))
                #tao = tao * (1 / (i + 1))
                w[int(y_i)] += x_i * tao
                w[y_hat] -= x_i * tao
    return w

# Calculate loss value.
def loss(x_i, w_yi, w_max_y):
    return max(0, 1 - np.dot(w_yi, x_i) + np.dot(w_max_y, x_i))

# Train SVM algorithm - return weight vectors.
def train_svm(train_x_data, train_y_data):
    w = np.zeros((3, 6))
    train_x_data = np.insert(train_x_data, 0, np.ones(train_x_data.shape[0]), axis=1)
    for i in range(97):
        for x_i, y_i in zip(train_x_data, train_y_data):
            d = np.dot(w, x_i)
            d[int(y_i)] = float('-inf')
            y_hat = np.argmax(d)
            eta = 0.1
            lambda_val = 0.0001
            w = (1 - lambda_val * eta) * w
            if loss(x_i, w[int(y_i)], w[int(y_hat)]) > 0:
                w[int(y_i)] += eta * x_i
                w[int(y_hat)] -= eta * x_i
    return w

# Get test exampels and weight vectors and return the predicted values.
def predict(test_x, w):
    test_y = []
    test_x = np.insert(test_x, 0, np.ones(test_x.shape[0]), axis=1)
    for i in range(len(test_x)):
        y_hat = np.argmax(np.dot(w, test_x[i]))
        test_y.append(y_hat)
    return test_y


train_x, train_y, test_x_file, outfile_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

train_x_data = np.loadtxt(train_x, delimiter=',')
train_y_data = np.loadtxt(train_y, delimiter='\n')
test_x = np.loadtxt(test_x_file, delimiter=',')

minmax_train_x = train_x_data.copy()
zscore_train_x = train_x_data.copy()
minmax_test_x = test_x.copy()
zscore_test_x = test_x.copy()

minmax(minmax_train_x, train_x_data)
zscore(zscore_train_x, train_x_data)
minmax(minmax_test_x, train_x_data)
zscore(zscore_test_x, train_x_data)


# K = 9 --> optimal.
knn_list = KNN(train_x_data, train_y_data, test_x, 9)

# Run all algorithms and save the results in lists.
w = train_perceptron(zscore_train_x, train_y_data)
per_list = predict(zscore_test_x, w)

w = train_pa(zscore_train_x, train_y_data)
pa_list = predict(zscore_test_x, w)

w = train_svm(zscore_train_x, train_y_data)
svm_list = predict(zscore_test_x, w)

# Write the results in outFile.
with open(outfile_name, "w") as f:
    for i in range(len(test_x)):
        f.write(f"knn: {knn_list[i]}, perceptron: {per_list[i]}, svm: {svm_list[i]}, pa: {pa_list[i]}\n")

