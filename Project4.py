
from scipy.sparse import data
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
# from scipy.spatial import distance_matrix


    
with open('Testdata4.txt',encoding="utf8") as file:
    test = file.readlines()  #list in element 
#Now removes /n and makes the string [1 2 3 4] into a list
arr =np.array([i.strip().split(',') for i in test])
# print(arr) #now all features are in one long array ['1 2 3 4'] time to split

arr = np.array([x.split(' ') for i in arr  #going through from list just made above seperate after each "space"
       for x in i]) #so now going through each iteration through "i" from list before
#now each of the 4 elements are seperated but still in STRING FROM
c = arr.astype(np.float64) #convert strings into back numeral values
# df = pd.DataFrame(c)
cleaned = preprocessing.normalize(c) # normalize data bw 1 and 0


#Now K-means,remember needs the ID clusters(3) splitting data into three sub clusters
#Have a function for kmeans
def kmeanF(dataSet,k,instances):
    # cleaned = dataSet.to_numpy()
    randPoints = np.random.choice(len(dataSet),k,replace=False) #choosing k points from dataTest and replace= false b/c unique
    cent = dataSet[randPoints] #making the 3 centriods
    cosineD = cdist(dataSet,cent,'cosine') #much better accru find distance between points and centriods
    #FIND CLOSEST 
    closestD =np.array([np.argmin(i) for i in cosineD])   #using argmin from will go through all 150 and find smallest distances from cosineD
    for i in range(instances):
        cent2 = []
        for r in range(k):
            cent1 = np.mean(dataSet[closestD==r],axis=0) #axis needed to print columns
            # Need to keep each iterate of updated centriod
            cent2.append(cent1)   
    newcosineD = cdist(dataSet,cent2,'cosine') #much better accru
    finalD =np.array([np.argmin(i) for i in newcosineD])
    # print(finalD)
    return finalD

k = 3
label = kmeanF(cleaned,k,150)
finalPoints= [x+1 for x in label] 
print(finalPoints)       


from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# visualizer = SilhouetteVisualizer(kmeanF, colors='yellowbrick')
# visualizer.fit(c)  
# visualizer.show()  

np.savetxt("miner4.txt",finalPoints, fmt= '%i')
