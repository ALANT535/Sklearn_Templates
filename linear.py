from sklearn import linear_model
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
boston_data=datasets.load_boston()

x=boston_data.data
y=boston_data.target

#506 training examples with 13 features in dataset

model=linear_model.LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model.fit(X_train,y_train)
predictions=model.predict(X_test)

print("\nThe accuracy is : ",model.score(x,y))

print("\ncoeffficients : ",model.coef_,type(model.coef_),np.shape(model.coef_),sep="\n")

plt.scatter(x.T[5],y)

plt.plot(predictions,y_test)
print("intercept:",model.intercept_)
    
#6 and 13