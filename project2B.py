import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# import pandas_ml as pdml
# from pandas_ml import confusion_matrix
  
data = pd.read_csv('trainB.csv')
dataTest = pd.read_csv('testB.csv')

#NUMERICAL OF TRAINING

encoderr = preprocessing.LabelEncoder()
data['sex'] = encoderr.fit_transform(data['sex'])  #this makes male = 1 and female = 0
data['age_cat'] = encoderr.fit_transform(data['age_cat'])  #this makes male = 1 and female = 0
# data['race'] = encoderr.fit_transform(data['race'])  #this makes male = 1 and female = 0
data['race'] = data['race'].replace('African-American',0)
data['race'] = data['race'].replace('Caucasian',1)
data['race'] = data['race'].replace('Hispanic',2)
data['race'] = data['race'].replace('Native American',3)
data['race'] = data['race'].replace('Asian',4)
data['race'] = data['race'].replace('Other',5)

data['c_charge_degree'] = encoderr.fit_transform(data['c_charge_degree'])  #this makes male = 1 and female = 0


#NUMERICAL Value FOR TEST
encoderT = preprocessing.LabelEncoder()
dataTest['sex'] = encoderT.fit_transform(dataTest['sex'])  #this makes male = 1 and female = 0
dataTest['age_cat'] = encoderT.fit_transform(dataTest['age_cat'])  #since there is three options(less,greater,25-45)0-2
dataTest['race'] = encoderT.fit_transform(dataTest['race'])  #this makes male = 1 and female = 0
dataTest['c_charge_degree'] = encoderT.fit_transform(dataTest['c_charge_degree'])  #this makes male = 1 and female = 0


label = data.iloc[:,-1]  #saves the last column whihch is the 0,1
#remove not needed columns and isn't numerical
data = data.drop(columns=['c_charge_desc','label'])
afTr = data.loc[data['race']==0]
white = data.loc[data['race']==1]
whiteLabel = label.iloc[:1733]
AfTLabel = label.iloc[:2580]
#data = data.drop(columns=['race','c_charge_desc','label'])
dataTest = dataTest.drop(columns=['c_charge_desc'])
#dataTest = dataTest.drop(columns=['race','c_charge_desc'])

normTr = preprocessing.normalize(data)
normTest = preprocessing.normalize(dataTest)
gb = GradientBoostingClassifier(learning_rate=0.1)
gb.fit(afTr,AfTLabel)
# predictt = gb.predict(normTest)
cross_predict = cross_val_predict(gb,afTr,AfTLabel,cv =5)
# cross_predictW = cross_val_predict(gb,white,whiteLabel,cv =5)
# scores = cross_val_score(gb,normTr,label,cv =5)
# print(cross_predict)
np.savetxt("minerB.txt",cross_predict, fmt = '%i')

X = afTr
Y = AfTLabel
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
gb1 = GradientBoostingClassifier(learning_rate=0.1)
gb1.fit(X_train,y_train) 
cp = cross_val_predict(gb1,afTr,AfTLabel,cv =5)
# cp = cross_val_predict(gb1,white,whiteLabel,cv =5)
y_test = label.iloc[:2580]
# y_test = label.iloc[:1733]
# acc = accuracy_score(y_test, cp)
# print(acc)
# print(normTr.shape)  5049 rows
confusionM=confusion_matrix(y_test, cp)  # to see all outcomes like FP,FN,TP,TN
print(confusionM)
[[tp , tn],[fn , fp]] =confusion_matrix(y_test, cp)
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)

#false positve rates fp = fp/fp+tn
fpr = fp/(fp+tn)
print(fpr)
ppv = tp/(tp+fp)







