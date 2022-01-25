
import enum
import numpy as np
import re
import nltk
from nltk.corpus.reader.chasen import test
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from numpy import vectorize
from numpy.core.fromnumeric import resize, sort  #for array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier


  
def cleanTag(rawhtml):
        cleanr=re.compile('<.*?>')
        cleantext=re.sub(cleanr, '', str(rawhtml)) #removes html tags
        res = re.sub(r'[^\w\s]', '', cleantext) #removes punct
        result = re.sub(r"\d+", "", res) #removes numbers
        return result

def stem_tokens(tokens, PSstemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(PSstemmer.stem(item))
    return stemmed

        
with open('TrainingD.txt',encoding="utf8") as file:
    train = file.readlines()  #list in element
print(train)

with open('TestD.txt',encoding="utf8") as file:
    testv = file.readlines()  #list in element

#Needed to hold thr sent
arr = []

for review in train:
    #string_name.split(separator, maxsplit) Need to split sentiment from the text
    revw = review.split("\t",1)
    arr.append(revw)
    
tr = cleanTag(train)
te = cleanTag(testv)
ps = PorterStemmer()
tokenTr = word_tokenize(tr)
tokenTe = word_tokenize(te)
#tokens_without_sw = [word for word in tokenTr if not word in stopwords.words()]
#tokens_without_sw1 = [word for word in tokenTe if not word in stopwords.words()]
stemmedafter = stem_tokens(tokenTr,ps)
stemmedafter1 = stem_tokens(tokenTe,ps)
#print(stemmedafter)


#Tfid vectorizing text more optimal than countizer
vectorizer = TfidfVectorizer()
#only use fit for one text file and transform for other
#MAKING TEXT INTO NUMERICAL VALUES
trainV = vectorizer.fit_transform(train)
testV = vectorizer.transform(testv)
#cosine similairty to find distance between the two texts if it matches closes to test review then its prob pos or neg
#rows is trainv and col is testv in the 15000 * 15000 matrix
cos = cosine_similarity(trainV,testV)
arrBefore = []
#(index,tuple)
#for count, val in enumerate(cos) :
    #print(count,val)
#print(res) 
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(trainV, testV)
#for t in range(len(trainV)):




  








