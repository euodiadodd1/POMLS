from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os
import pandas as pd

random.seed(1)
np.random.seed(1)
NUM_USERS = 20 # should be muitiple of 10
NUM_LABELS = 2
# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml('mnist_784', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []
for i in trange(10):
    idx = mnist.target==str(i)
    mnist_data.append(mnist.data[idx])

combined = list(zip(mnist_data, range(10)))

user_samples = []


user_labels = []

#randomly assign labels
for i in range(NUM_USERS):
    l = np.random.randint(low=0, high=10, size=2)
    user_labels.append(l)

labels, counts = np.unique(user_labels, return_counts=True)

#Randomly split data for clients

for i,x in enumerate(mnist_data):
    I = counts[i]
    result = np.array_split(x, I)
    user_samples.append(result)



# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    count = 0
    uname = 'f_{0:05d}'.format(i)
    train_data['users'].append(uname) 
    X = pd.DataFrame()
    y = []
    for j in range(10):
        if j in user_labels[i]:
            sample_X = user_samples[j][count]
            #X[i][:], y[i][:] = zip(*combined)
            num_samples = len(sample_X) 
            sample_y = num_samples*[j]
            train_len = int(0.75*num_samples)
            test_len = num_samples - train_len
            X = pd.concat([X, sample_X])
            y = np.concatenate([y,sample_y])
            count += 1
    
    print(x.shape)
    #print("x:",X[:][:train_len].values.tolist())
    print("y:", y[:][:train_len])
    train_data['user_data'][uname] = {'x': X[:][:train_len].values.tolist(), 'y': y[:][train_len:].tolist()}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[:][train_len:].values.tolist(), 'y': y[train_len:].tolist()}
    test_data['num_samples'].append(test_len)
   


print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
