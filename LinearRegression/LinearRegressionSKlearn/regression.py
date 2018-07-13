import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# Read the Data
df = quandl.get("WIKI/GOOGL", api_key="GsVohghtJWdB8vYD8_Kr")

# Only Keep Relevant Data
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Do Calculations with Data to make new columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Keep Relevant Data
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', "Adj. Volume"]]

# Variable to see which column to read from
forecast_col = 'Adj. Close'
# Calculates shift of how far into the future we are shifting all the data
forecast_out = int(math.ceil(0.01*len(df)))
print '\n\nlooking ' + str(forecast_out) + ' days into the future'

# Deal with stuff that is not a number
df.fillna(-99999, inplace=True)

# Actually make shift
df['label'] = df[forecast_col].shift(-forecast_out)

# Deal with stuff that is not a number
df.dropna(inplace=True)

# Setting x to everything but the label col and y into the label col
x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Scaling x down; Can skip this step
# Especially skip if working with lots of new data (like getting new data everyday)
# since new data must scaled the same as training data
x = preprocessing.scale(x)

# Shuffles and then splits the data into training x and y and testing x and y
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# Create classifier for linear regression
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
clf = LinearRegression(n_jobs=-1)

# n_jobs=1 --> default
# n_jobs=10 --> will make it thread 10 times to make the training time faster
# n_jobs=-1 --> will do the maximum amount of threads to optimize training time
# clf = LinearRegression(n_jobs=-1)


# Create classifier for support vector linear algorithm
# clf = svm.SVR()
# Train the new Data into the Classifier; fit --> train
clf.fit(x_train, y_train)
# Test some data and returns the accuracy; score --> test
accuracy = clf.score(x_test, y_test)
print accuracy
