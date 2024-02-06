# -*- coding: utf-8 -*-


"""
Created on Tue Jan 30 14:17:10 2024

@author: 91915
"""
import pandas as pd
import numpy as np
wbcd = pd.read_csv("C:/1-Python/2-dataset/wbcd.csv.xls")
wbcd
#this dataset has 569 rows & 32 columns
#to understand the distribution of data
wbcd.describe()

wbcd.info()
#this dataset not contains any null values

wbcd['diagnosis'].value_counts()
#B    357
#M    212

#in output if there is B replace it with Beniegn and
#replace M with Maligant
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='B','Beniegn',wbcd['diagnosis'])
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Maligant',wbcd['diagnosis'])

#patient ID is not important so drop that column
wbcd = wbcd.drop('id',axis=1)

#normalization
def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#let us apply normlization function to the dataset
wbcd_norm = norm(wbcd.iloc[:,1:32])


#now let us take X as input & Y as output
X = np.array(wbcd_norm.iloc[:,1:32])
Y = np.array(wbcd['diagnosis'])

#split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
#to avoid the unbalancing of data during splitting the concept of 
#statified sampling is used

#now build the KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
pred

#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
pd.crosstab(pred,Y_test)
#here for 5 patients that are Beneign but model predicted as Maligant 
#so this model is not accepted

#let us try to select the correct value of k
acc= []
#running the KNN algorithm for k=3 to 50 in step of 2
#k's value is selected as odd
for i in range(3,50,2):
    #declare model
    n = KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train,Y_train)
    train_acc = np.mean(n.predict(X_train) == Y_train)
    test_acc = np.mean(n.predict(X_test) == Y_test)
    acc.append([train_acc,test_acc])

#lets plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'bo-')
#there are valiues like 3,5,7,9 where the accuracy is good

#lets try for K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
accuracy_score(pred,Y_test)
pd.crosstab(pred,Y_test)
#here we get 3 wrong predictions