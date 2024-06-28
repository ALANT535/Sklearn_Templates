import pandas as pd
import numpy as np
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

s=pd.read_csv(r"C:\Users\LENOVO\Documents\Important_documents\VIT\sem2\python\vsc\car.data")

x=s[["buying",
    "maint",
    "safety"]].values

y=s[['class']]

le=LabelEncoder()

for i in range(len(x[0])):
    x[:,i]=le.fit_transform(x[:,i])
    


label_mapping={
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3
}


y['class']=y['class'].map(label_mapping)
y=np.array(y)
knn=neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

knn.fit(X_train,y_train)

predictions=knn.predict(X_test)

accuracy=metrics.accuracy_score(y_test,predictions)
# print(predictions,end="\n\n")

a=1102

# print(y)

print("\n\nThe accuracy is ", accuracy)
print("\n\nprediction is :",knn.predict(x)[a])
print("\nactual value is : ",y[a])
