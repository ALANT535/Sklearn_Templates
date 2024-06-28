from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import svm

iris=datasets.load_iris()
dat=iris.data
pred=iris.target
X_train,X_test,y_train,y_test=train_test_split(dat,pred,test_size=0.2)


model=svm.SVC()

model.fit(X_train,y_train)

predi=model.predict(X_test)

accuracy=metrics.accuracy_score(predi,y_test)

classes=['Iris setosa','Iris Versicolour','Iris Virginica']

print("The accuracy of model is : ",accuracy)
a=29
actual_predictions,actual_values=[],[]

for i in predi:
    actual_predictions.append(classes[i])
    
for j in y_test:
    actual_values.append(classes[j])


predi,y_test=actual_predictions,actual_values


print("actual value is :",y_test,end="\n\n\n")
print("predicted value is : ",predi)    

    
    