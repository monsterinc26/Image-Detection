# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:40:03 2020

@author: Ravi
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

image1=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 1.jfif')
gray1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
t=np.resize(gray1,(60,60))
y=np.reshape(t,(1,3600))
dataa=pd.DataFrame(y)
plt.imshow(t)

image2=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 2.jfif')
gray2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
new_image=np.resize(gray2,(60,60))
a=np.reshape(new_image,(1,3600))
dataa1=pd.DataFrame(a)

image3=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 3.jfif')
gray3=cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
new_image1=np.resize(gray3,(60,60))
b=np.reshape(new_image1,(1,3600))
dataa2=pd.DataFrame(b)

image4=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 4.jfif')
gray4=cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)
new_image2=np.resize(gray4,(60,60))
c=np.reshape(new_image2,(1,3600))
dataa3=pd.DataFrame(c)

image5=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 5.jfif')
gray5=cv2.cvtColor(image5,cv2.COLOR_BGR2GRAY)
new_image3=np.resize(gray5,(60,60))
d=np.reshape(new_image3,(1,3600))
dataa4=pd.DataFrame(d)

image6=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 6.jfif')
gray6=cv2.cvtColor(image6,cv2.COLOR_BGR2GRAY)
new_image4=np.resize(gray6,(60,60))
e=np.reshape(new_image4,(1,3600))
dataa5=pd.DataFrame(e)

image7=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 7.jfif')
gray7=cv2.cvtColor(image7,cv2.COLOR_BGR2GRAY)
new_image5=np.resize(gray7,(60,60))
f=np.reshape(new_image5,(1,3600))
dataa6=pd.DataFrame(f)

image8=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 8.jfif')
gray8=cv2.cvtColor(image8,cv2.COLOR_BGR2GRAY)
new_image6=np.resize(gray8,(60,60))
g=np.reshape(new_image6,(1,3600))
dataa7=pd.DataFrame(g)

image9=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 9.jfif')
gray9=cv2.cvtColor(image9,cv2.COLOR_BGR2GRAY)
new_image7=np.resize(gray9,(60,60))
h=np.reshape(new_image7,(1,3600))
dataa8=pd.DataFrame(h)

image10=cv2.imread('C:/Users/Ravi/Downloads/dogs/dogs 10.jfif')
gray10=cv2.cvtColor(image10,cv2.COLOR_BGR2GRAY)
new_image8=np.resize(gray10,(60,60))
i=np.reshape(new_image8,(1,3600))
dataa9=pd.DataFrame(i)

dataa=dataa.append([dataa1,dataa2,dataa3,dataa4,dataa5,dataa6,dataa7,dataa8,dataa9])

path='C:/Users/Ravi/Downloads/cats'
da=pd.DataFrame()
for i in os.listdir(path):
    
    image=cv2.imread(path+'/'+i)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    new_imagee=np.resize(gray_image,(60,60))
    f=np.reshape(new_imagee,(1,3600))
    da=da.append(list(f))
    print(da)
    
da=da.append([dataa])    
dataset=da.to_csv('D:\python\Open Cv2\dogsandcats.csv')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

data=pd.read_csv('D:/python/Open Cv2/dogsandcats.csv')
data.info()
data.describe()

test=cv2.imread('C:/Users/Ravi/Downloads/cats/cat test.jfif')
gray=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
f=np.resize(gray,(60,60))
new=np.reshape(f,(1,3600))

x=data.iloc[:,1:-1]
y=data.iloc[:,-1:]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

svc=SVC()
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
svc.predict(new)
svc.score(x,y)

log=LogisticRegression()
log.fit(x_train,y_train)
log_pred=log.predict(x_test)
log.predict(new)
log.score(x,y)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn_pred=knn.predict(x_test)
b=knn.predict(new)
knn.score(x,y)

from sklearn.metrics import mean_squared_error 
error=mean_squared_error(y_test,pred)
log_error=mean_squared_error(y_test,pred)

t=1-int(knn.score(x,y))















































