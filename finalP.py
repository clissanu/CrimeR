import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('FINALDATA1.txt',sep = ",", header =None)
df.columns = ['age', 'sex','cp','tp', 'chol','fbs','r','thal','exa','oldpeak','slope','ca','thal1','label']
df['label'] = df['label'].where(df['label'] == 0, 1) #changes non zeros to 1 because it is not needed 2,3,4
label = df.iloc[:,-1]  #saves the last column which is the 0,1
data = df.drop(columns=['oldpeak','slope','label']) #since label is saved no need to have it in main df
normData = preprocessing.normalize(data)

#time to answer question is it fair upon gender first make sure input data match
#normData -> having all males in one var and all female values in another var
#now to focus on the sex(protected attr)
male = data.loc[data['sex']==1]  #having ONLY male info... rows ->206
#error because the rows dont match with label
labelM = label.iloc[:206] #this will match up with 206 rows in male will fix cp error
female = data.loc[data['sex']==0]  #having ONLY female values.... rows ->97
# print(len(female.index))
labelF = label.iloc[:97] #this will match up with 97 rows in female

#FIRST FIND ACCURACY NO SPLIT NEEDED
gb = GradientBoostingClassifier(learning_rate=0.1)
# gb.fit(female,labelF)
gb.fit(male,labelM)
# gb.fit(normData,label)
#cross_predict = cross_val_predict(gb1,male,label,cv =5)#Found input variables with inconsistent numbers of samples: [206, 303]
# cp = cross_val_predict(gb,normData,label,cv =5) #BETTER male rows match with NEW male label
cp = cross_val_predict(gb,female,labelF,cv =5)
# cp = cross_val_predict(gb,male,labelM,cv =5)
# conf=confusion_matrix(label, cp) to see all outcomes like FP,FN,TP,TN
[[tp , tn],[fn , fp]] = confusion_matrix(labelF, cp)
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)
# print("FN Respectively, that is: ",  round(fn/(fn+tp) * 100), "%")
print("CAL For gradient boosting: ",  round(tp/(fp+tp) * 100), "%")

acc = accuracy_score(labelF,cp)  #acc is about %75 for NORMAL data
print(acc)



#trying rf
rf1 = RandomForestClassifier(n_estimators = 1000)
rf1.fit(female,labelF)
#print(x)
cross =cross_val_predict(rf1,female,labelF,cv =5)
[[tp , tn],[fn , fp]] = confusion_matrix(labelF, cross)
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)
# print("False negatives : ", fn, " Respectively, that is: ",  round(fn/(fn+tp) * 100), "%")
print("CAL Respectively , that is: ",  round(tp/(fp+tp) * 100), "%")
acc = accuracy_score(labelF,cross)  #acc is about %75 for NORMAL data
print(acc)
#this is just the GB WITHOUT CROSS VAL
# #X = normData
# Y = label
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
# gb1 = GradientBoostingClassifier(learning_rate=0.1)
# gb1.fit(X_train,y_train) 
# predictofy = gb1.predict(X_test)