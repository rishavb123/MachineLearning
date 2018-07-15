import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups! ')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            distances.append([euclidean_distance, group])

    votes = [dis[1] for dis in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]

    c = float(Counter(votes).most_common(1)[0][1])/k

    # print vote_result, c

    return vote_result, c


df = pd.read_csv('../data/breast-cancer-wisconsin.data.csv')

df.replace('?', -99999, inplace=True)
df.drop('id', 1, inplace=True)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for classification in test_set:
    for datap in test_set[classification]:
        vote, confidence = k_nearest_neighbors(train_set, datap, k=5)
        if classification == vote:
            correct += 1
        # else:
        #     print confidence
        total += 1

print 'Accuracy: ', float(correct)/total
