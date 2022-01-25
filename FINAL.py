import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification


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

gb = GradientBoostingClassifier(learning_rate=0.1)
gb.fit(female,labelF)
# cp = cross_val_predict(gb,female,labelF,cv =5)
cp = cross_val_predict(gb,female,labelF,cv =5)
conf=confusion_matrix(labelF, cp)
acc = accuracy_score(labelF,cp)
print(acc)
# print(conf)


#NEEDED ACC TO SEE HOW ACCURATE MODEL IS 
X = normData
Y = label
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
gb1 = GradientBoostingClassifier(learning_rate=0.1)
gb1.fit(X_train,y_train) 
predictofy = gb1.predict(X_test)
#now using cross valid to eval to see how well our model works on data it hasnt seen before
# cross_predict = cross_val_predict(gb1,X,label,cv =5)
#cross_predict = cross_val_predict(gb1,male,label,cv =5)#Found input variables with inconsistent numbers of samples: [206, 303]
cross_predict = cross_val_predict(gb1,male,labelM,cv =5)#BETTER male rows match with NEW male label
# print(cross_predict)
# acc = accuracy_score(y_test, predictofy)
# print(acc)


#Now to find opportunity cost bias(GENDER) using false negative rates
cm=confusion_matrix(labelM, cross_predict)  # to see all outcomes like FP,FN,TP,TN
print(cm)
[[tp , tn],[fn , fp]] =confusion_matrix(labelM, cross_predict) #find values for MALES
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)








print('Number of rows in Dataframe : ', len(data.index))
print('Number of colums in Dataframe : ', len(data.columns))

np.savetxt("finalT.txt",df, fmt = '%s') #to see dataframe
