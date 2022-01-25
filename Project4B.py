import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from sklearn import preprocessing 
import umap.umap_ as umap
from scipy.spatial.distance import cdist 

# with open('new_test.txt',encoding="utf8") as file:
#         test = file.readlines()  #list in element
# arr =np.array([i.strip().split(',') for i in test])
# # c = arr.astype(np.int64)
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('new_test.txt',sep = ",", header =None)
# df = pd.DataFrame(c)

df = df.loc[:, (df != 0).any(axis=0)] #remove the ALL ZEROS COL
removeC = np.where(df != 0,1,df)
# removeC = preprocessing.normalize(df) # MAKE ALL 0s and 1s
# print(removeC)
# pca = PCA(n_components=2)
# pca.fit(removeC)
umapp = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,).fit_transform(removeC)

#Now K-means,remember needs the ID clusters(3) splitting data into three sub clusters
#Have a function for kmeans
def kmeanF(dataSet,k,instances):
    # cleaned = dataSet.to_numpy()
    randPoints = np.random.choice(len(dataSet),k,replace=False) #choosing k points from dataTest and replace= false b/c unique
    cent = dataSet[randPoints]
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



from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# visualizer = SilhouetteVisualizer(kmeanF, colors='yellowbrick')
# visualizer.fit(removeC)  
# visualizer.show()  
k = 10    
label = kmeanF(umapp,k,500)
label = [x+1 for x in label]
# print(label)
np.savetxt("miner4B.txt",label, fmt= '%i')