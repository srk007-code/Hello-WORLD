# ML Project
# Diabetes Detection

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('datasets_228_482_diabetes.csv')

data.shape

data.head()

data.isnull().values.any()

cormat = data.corr()

top_corr_feat = cormat.index

plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_feat].corr(),annot = True,cmap = 'RdYlGn')

data.corr()

dia_True_counts = len(data.loc[data['Outcome']==1])
dia_False_counts  = len(data.loc[data['Outcome']==0])
(dia_True_counts,dia_False_counts)

from sklearn.model_selection import train_test_split
X = data.drop('Outcome',axis=1).values
y = data['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['BMI'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['Age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['SkinThickness'] == 0])))

from sklearn.impute import SimpleImputer
fill_Values  = SimpleImputer(missing_values=0,strategy='mean')

X_train = fill_Values.fit_transform(X_train)
X_test = fill_Values.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
randome_forest_model = RandomForestClassifier(random_state=10)
randome_forest_model.fit(X_train,y_train.ravel())

predictions = randome_forest_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))













