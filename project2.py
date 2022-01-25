from nltk.featstruct import _trace_unify_identity
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# with open('train2.csv', encoding="utf8") as file:
#     train = file.readlines()  #list in element

# with open('test2.csv', encoding="utf8") as file:
    #test = file.readlines()  #list in element


data = pd.read_csv('train2.csv',header=None)
dataTest = pd.read_csv('test2.csv',header=None)
#first normalize the training and test from [0,1]
norm = preprocessing.normalize(data)
normT = preprocessing.normalize(dataTest)
norm = np.delete(norm, -1, axis = 1)  #now this is new train with 10 features b/c I was getting an error

result = data.iloc[:,-1]  #saves the last column whihch is the 0,1
#print(result)
decision = DecisionTreeClassifier()  #creating decision treeObj
dTrain = decision.fit(norm,result)  #so now train on the normalzied data and results
trainp = dTrain.predict(normT)
#print(trainp)
np.savetxt("miner1.txt",trainp, fmt = '%i')


# normD = preprocessing.normalize(data)
# normD = np.delete(normD, -1, axis = 1)
#fl score changes each time due to the 80->20 split begin random each time
#print(type(normD))
#print(normD)
X = norm
Y = result
#first splitting the data training 80% test 20%  needed fro f1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
dT = DecisionTreeClassifier()  #creating decision treeObj

#doing the actual training on the training set(80)
dt = dT.fit(X_train,y_train)

#predicts spits what model "thinks" that data is y test(is actual score) x_test is 20
trainPred = dt.predict(X_test)


#F1 SCORE LOOK AT PP HIGHER F1 THE BETTER NEVER CAN BE ACC
flScore = f1_score(y_test, trainPred)
print(flScore)





