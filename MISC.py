import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from linear import X_train

data=datasets.load_breast_cancer()

dat=scale(data.data)
labels=data.target

x_train,x_test,y_train,y_test=train_test_split(dat,labels,test_size=0.2)
model=KMeans(n_clusters=2,random_state=0)

model.fit(X_train,y_train)

predictions=model.predict(x_test)

accuracy=metrics.accuracy_score(predictions,y_test)
# print("bruh cmon: ",accuracy)


print("hello is this even working")