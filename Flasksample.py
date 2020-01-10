#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:33:51 2019

@author: beast
"""


#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


pd.set_option('display.max_columns',30) # set the maximum width

# Load the dataset in a dataframe object 
df = pd.read_csv('/Users/data.csv')

# Explore the data check the column values
print(df.columns.values)
print(df.head())
categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.dtypes

#check for null values
print(len(df) - df.count())  #Cabin , boat, home.dest have so many missing values


include = ['radius_mean','texture_mean', 'perimeter_mean', 'area_mean','compactness_mean','symmetry_mean']
df_ = df[include]
print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['radius_mean'].unique()
df_['texture_mean'].unique()
df_['perimeter_mean'].unique()
df_['area_mean'].unique()

# check the null values
print(df_.isnull().sum())
print(df_['radius_mean'].isnull().sum())
print(df_['texture_mean'].isnull().sum())
print(len(df_) - df_.count())

df_.dropna(axis=0,how='any',inplace=True)  

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
print(categoricals)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
pd.set_option('display.max_columns',30)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())


from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['radius_mean'].describe())
print(scaled_df['texture_mean'].describe())

print(scaled_df['perimeter_mean'].describe())
print(scaled_df['area_mean'].describe())
print(scaled_df.dtypes)


from sklearn.linear_model import LogisticRegression
dependent_variable = 'diagnosis'
# Another way to split the three features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = dependent_variable
#convert the class back into integer
y = df.iloc[:,1].values


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))



# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

testY_predict = lr.predict(testX)
testY_predict.dtype
print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

import joblib 
joblib.dump(lr, '/Users/beast/model.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/beast/model_columns.pkl')
print("Models columns dumped!")

#Count the number of rows and columns in the data set
df.shape

#print details statistics
df.describe

#Count the empty (NaN, NAN, na) values in each column
df.isna().sum()


#Get a count of the number of 'M' & 'B' cells
df['radius_mean'].value_counts()

#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")

#Look at the data types 
df.dtypes

#Encoding categorical data values (
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))

sns.pairplot(df, hue="diagnosis")
sns.pairplot(df.iloc[:, 1:6], hue="diagnosis")

df.head(5)

#Get the correlation of the columns
df.corr()

plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True, fmt='.0%')

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
model = forest.fit(X_train, Y_train)


X = df.iloc[:, 2:8].values 
Y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


  #Check precision, recall, f1-score
print( classification_report(Y_test, model.predict(X_test)) )
  #Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line
print(X_test)
    
#Print Prediction of Random Forest Classifier model
pred = model.predict(X_test)
print(pred)
#Print a space
print()
pred = model.predict([[0.323872,0.007066,0.119205,0.051019,0.003738,0.499316]])
print(pred)
print()
#Print the actual values
print(Y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train,Y_train):
  
  #Using Logistic Regression 
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier 
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC linear
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC rbf
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB 
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier 
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest

model = models(X_train,Y_train)


from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))
  
  TN = cm[0][0]
  TP = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]
  
  print(cm)
  print('Model[{}] Testing Accuracy = "{}!"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
  print()# Print a new line

plt.hist(df['radius_mean'])

plt.show()
