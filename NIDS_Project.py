# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading data
train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")

# checking the dimensions of the datasets

print(f"Training :: {train.shape}")
print(f"Testing :: {test.shape}")

train.head()

train.info()

test.head()

### Data Analysis

# statistical summary 

train.describe()

train.describe(include='object')

# unique types of flag

flag = train['flag'].unique()
flag

plt.figure(figsize=(15, 3))
values = train['flag'].value_counts()
plt.bar(flag, values)

# unique types of protocol_type

protocol_types = train['protocol_type'].unique()
protocol_types

plt.figure(figsize=(10, 5))
values = train['protocol_type'].value_counts()
plt.bar(protocol_types, values)

# most used data : tcp, then udp, then icmp

# unique types of service

service = train['service'].unique()
service

plt.figure(figsize=(15, 3))
values = train['service'].value_counts()
plt.bar(service, values)

# unique types of class
class_attack = train['class'].unique()
class_attack

# Missing Values
missing_values = train.isnull().sum()
missing_values

# Checking for any duplicates
print(f"No. of duplicate rows ::  {train.duplicated().sum()}")

# dropping redundant columns in both train and test set
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# attack distribution
train["class"].value_counts()

class_type = train['class'].unique()
plt.figure(figsize=(7, 5))
values = train['class'].value_counts()
plt.bar(class_type, values)

#### Data Preprocessing

# feature scaling => scale cols numerical values to have 0 or 1 values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

col = train.select_dtypes(include=['int64', 'float64']).columns
cols = test.select_dtypes(include=['int64', 'float64']).columns

sc_train = scaler.fit_transform(train.select_dtypes(include=['int64', 'float64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['int64', 'float64']))

std_train = pd.DataFrame(sc_train, columns=col)
std_test = pd.DataFrame(sc_test, columns=cols)

std_train.head()

std_test.head()

# One-Hot Ecoding => dealing with categorical values
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# extracting categorical variables from both train and test datasets
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

# encoding categorical values
en_train = cattrain.apply(encoder.fit_transform)
en_test = cattest.apply(encoder.fit_transform)

en_Ytrain = en_train[['class']].copy()

# drop the target column => class
en_train = en_train.drop(['class'], axis=1)

##### Categorical values in train dataset before and after encoding
print(cattrain.head())  # categorical data before encoding
print('--------------------')
print(en_train.head())  # encoded categorical data

###### Categorical values in test dataset before and after encoding
print(cattest.head())  # categorical data before encoding
print('--------------------')
print(en_test.head())  # encoded categorical data

# Join the preprocessed categorical and numeric values

# train
train_X = pd.concat([std_train, en_train], axis=1)
train_y = en_Ytrain

# test
test = pd.concat([std_test, en_test], axis=1)

train_X.head()

train_y.head()

test.head()

### MODEL

# Split The Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_y, train_size=0.70,
                                                    random_state=2)

X_train.head(3)

Y_train.head(3)

X_test.head(5)

Y_test.head(5)


#### FITTING THE MODEL

# Data => train_X, train_y, test

# import libraries
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import time

# Train KNeighborsClassifier Model
KNN_classifier = KNeighborsClassifier()
start_time = time.time()
KNN_classifier.fit(X_train, Y_train)
end_time = time.time()
KNN_train_time = end_time - start_time
print("Training time :: ", KNN_train_time)

# Testing time for KNN
start_time = time.time()
y_test_pred = KNN_classifier.predict(X_train)
end_time = time.time()
KNN_test_time = end_time - start_time
print("Testing time: ", KNN_test_time)

# Train LogisticRegression Model
LGR_classifier = LogisticRegression(random_state=0)
start_time = time.time()
LGR_classifier.fit(X_train, Y_train)
end_time = time.time()
LGR_train_time = end_time - start_time
print("Training time :: ", LGR_train_time)

# Testing time for LGR
start_time = time.time()
y_test_pred = LGR_classifier.predict(X_train)
end_time = time.time()
LGR_test_time = end_time - start_time
print("Testing time: ", LGR_test_time)

# Train Gaussian Naive Baye Model
BNB_classifier = BernoulliNB()
start_time = time.time()
BNB_classifier.fit(X_train, Y_train)
end_time = time.time()
BNB_train_time = end_time - start_time
print("Training time :: ", BNB_train_time)

# Testing time for Naive Baye
start_time = time.time()
y_test_pred = BNB_classifier.predict(X_train)
end_time = time.time()
BNB_test_time = end_time - start_time
print("Testing time: ", BNB_test_time)

# Train Decision Tree Model
DTC_classifier = tree.DecisionTreeClassifier(random_state=0)
start_time = time.time()
DTC_classifier.fit(X_train, Y_train)
end_time = time.time()
DTC_train_time = end_time - start_time
print("Training time :: ", DTC_train_time)

# Testing time for DT
start_time = time.time()
y_test_pred = DTC_classifier.predict(X_train)
end_time = time.time()
DTC_test_time = end_time - start_time
print("Testing time: ", DTC_test_time)

# Train Random Forest Model
from sklearn.ensemble import RandomForestClassifier

RFC_classifier = RandomForestClassifier(n_estimators=30)
start_time = time.time()
RFC_classifier.fit(X_train, Y_train)
end_time = time.time()
RFC_train_time = end_time - start_time
print("Training time: ", RFC_train_time)

# Testing time for RF
start_time = time.time()
y_test_pred = RFC_classifier.predict(X_train)
end_time = time.time()
RFC_test_time = end_time - start_time
print("Testing time: ", RFC_test_time)

# Training Time
names = ['KNN', 'LR', 'NB', 'DT', 'RF']
values = [KNN_train_time, LGR_train_time, BNB_train_time, DTC_train_time, RFC_train_time]
plt.figure(figsize=(10, 5), num=20)
plt.bar(names, values)

# Testing Time
names = ['KNN', 'LR', 'NB', 'DT', 'RF']
values = [KNN_test_time, LGR_test_time, BNB_test_time, DTC_test_time, RFC_test_time]
plt.figure(figsize=(10, 5), num=20)
plt.bar(names, values)

#### EVALUATING THE MODEL  --  train data

from sklearn import metrics

models = []
models.append(('KNeighborsClassifier', KNN_classifier))
models.append(('LogisticRegression', LGR_classifier))
models.append(('Naive Baye Classifier', BNB_classifier))
models.append(('Decision Tree Classifier', DTC_classifier))
models.append(('Random Forest Classifier', RFC_classifier))

for i, val in models:
    scores = cross_val_score(val, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, val.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, val.predict(X_train))
    classification = metrics.classification_report(Y_train, val.predict(X_train))

    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

#### VALIDATING THE MODEL -- test data

for i, val in models:
    accuracy = metrics.accuracy_score(Y_test, val.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, val.predict(X_test))
    classification = metrics.classification_report(Y_test, val.predict(X_test))
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

# Training Accuracy
names = ['KNN', 'LR', 'NB', 'DT', 'RF']
values = [99.38, 95.52, 90.72, 100.0, 100.0]
plt.figure(figsize=(10, 5), num=20)
plt.bar(names, values)

# Testing Accuracy
names = ['KNN', 'LR', 'NB', 'DT', 'RF']
values = [99.17, 95.51, 90.67, 99.39, 99.68]
plt.figure(figsize=(10, 5), num=20)
plt.bar(names, values)

#### PREDICTING => using the test dataset

# PREDICTING FOR TEST DATA using KNN
knn_pred = KNN_classifier.predict(test)
NB_pred = BNB_classifier.predict(test)
log_pred = LGR_classifier.predict(test)
dt_pred = DTC_classifier.predict(test)
rf_pred = RFC_classifier.predict(test)

# Testing for first row
for i, val in models:
    print("For model: ", i)
    print("Expected: ", Y_test.iloc[0], "Predicted: ", val.predict(X_test).reshape(1, -1)[0][0])
    print()

# Testing for second row
for i, val in models:
    print("For model: ", i)
    print("Expected: ", Y_test.iloc[1], "\tPredicted: ", val.predict(X_test).reshape(1, -1)[0][1])
    print()

# Testing for five row
for i, val in models:
    print("For model: ", i)
    print("Expected: ", Y_test.iloc[4], "\tPredicted: ", val.predict(X_test).reshape(1, -1)[0][4])
    print()

# Testing for row 10
for i, val in models:
    print("For model: ", i)
    print("Expected: ", Y_test.iloc[9], "\tPredicted: ", val.predict(X_test).reshape(1, -1)[0][9])
    print()

# Testing for random rows
random_rows = np.random.randint(len(Y_test), size=(5))

for i, val in models:
    for j in random_rows:
        print("For model: ", i)
        print("Expected: ", Y_test.iloc[j], "\tPredicted: ",
              val.predict(X_test).reshape(1, -1)[0][j])
        print()

# Testing for random rows
random_rows = np.random.randint(len(Y_test), size=(5))

for j in random_rows:
    for i, val in models:
        print("For model: ", i)
        print("Expected: ", Y_test.iloc[j], "\tPredicted: ",
              val.predict(X_test).reshape(1, -1)[0][j])
        print()

# locating a row given the value
# locatedRow = Y_test.loc[Y_test['class'] == 2900]
# print(locatedRow)


### VOTING CLASSIFIER - ensemble

# import libraries
from sklearn.ensemble import VotingClassifier

# # Creating the ensemble model
# def ensembleModel(df_trainX, df_trainY, df_testX, df_testY):

# voting 'hard' - majority vote based on individual models
ensemble_model = VotingClassifier(estimators=
[
    ('KNN - ', KNN_classifier),
    ('LGR - ', LGR_classifier),
    ('BNB - ', BNB_classifier),
    ('DT - ', DTC_classifier),
    ('RF', RFC_classifier)
],
    voting='hard')

# Fitting the model on the training data

import pickle

trainned_model = ensemble_model.fit(X_train, Y_train)

pickle.dump(trainned_model, open('trainned_model.pkl', 'wb'))

# # Predicting on the testing data
# y_pred = ensemble_model.predict(df_testX)

# # Testing for random rows
# random_rows = np.random.randint(len(Y_test), size = (5))

# for j in random_rows:
#     print ("Expected: ", df_testY.iloc[j], "\tPredicted: ",val.predict(df_testX).reshape(1, -1)[0][j] )
#     print()

# # Evaluating the accuracy of the model
# accuracy = metrics.accuracy_score(df_testY, val.predict(df_testX))
# print(f"\nAccuracy: {accuracy}")

# return ensemble_model


# import streamlit as st
# # create a title and description for the app
# st.title('Network Intrusion Detection')
# st.write('This system helps to detect network intrusions by monitoring network traffic in real-time.')

# # create a text input field for entering the IP address or port number to monitor
# ip_address = st.text_input('Enter the IP address or port number to monitor')

# # create a checkbox to enable/disable real-time monitoring
# real_time = st.checkbox('Enable real-time monitoring')

# # create a button to start/stop monitoring
# if st.button('Start/Stop Monitoring'):
#     if real_time:
#         # start monitoring network traffic
#         st.write('Monitoring network traffic...')
#         # code to monitor network traffic in real-time
#     else:
#         # stop monitoring network traffic
#         st.write('Stopped monitoring network traffic.')

# # create a table to display network traffic data
# traffic_data = pd.read_csv('Test_data.csv')
# # [
# #     {'Time': '12:00 PM', 'Source IP': '192.168.1.1', 'Destination IP': '192.168.1.2',
# #      'Protocol': 'TCP', 'Status': 'Successful'},
# #     {'Time': '12:05 PM', 'Source IP': '192.168.1.3', 'Destination IP': '192.168.1.4',
# #      'Protocol': 'UDP', 'Status': 'Failed'},
# #     {'Time': '12:10 PM', 'Source IP': '192.168.1.5', 'Destination IP': '192.168.1.6',
# #      'Protocol': 'TCP', 'Status': 'Successful'},
# # ]

# st.write('Network Traffic Data:')
# #st.table(traffic_data.head())

# # create a plot to visualize network traffic
# st.write('Network Traffic Visualization:')
# st.line_chart([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# # create a section to display intrusion alerts
# intrusion_alerts = [
#     {'Time': '12:05 PM', 'Source IP': '192.168.1.3', 'Alert': 'Unauthorized access detected'},
#     {'Time': '12:20 PM', 'Source IP': '192.168.1.7', 'Alert': 'Malware detected'},
# ]

# st.write('Intrusion Alerts:')
# for alert in intrusion_alerts:
#     st.write(alert['Time'], '-', alert['Source IP'], '-', alert['Alert'])
