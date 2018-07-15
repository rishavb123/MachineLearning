import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Use comments from Linear Regression

# Read in data
df = pd.read_csv('breast-cancer-wisconsin.data.csv')

df.replace('?', -99999, inplace=True)
# df.dropna(inpace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print accuracy

# example values from doc excluding id
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print prediction
