# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SASIDEVI V
RegisterNumber:  212222230136
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data =np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
Y=data[:,2]

X[:5]

Y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)


def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```


## Output:
### Array value of x
![ML51](https://user-images.githubusercontent.com/118707332/235416662-2bf0d07d-2206-404c-ab6f-5f9287ce23b5.png)
### Array Value of y
![ML52](https://user-images.githubusercontent.com/118707332/235416668-a896a855-5f69-497e-939c-222d6e1ef9c1.png)
### Exam 1- Score graph
![ML53](https://user-images.githubusercontent.com/118707332/235416677-2cf55685-3b4a-4683-ac30-2872b8bc4f6c.png)
### Sigmoid Function Graph
![ML54](https://user-images.githubusercontent.com/118707332/235416691-6e7e275c-c5d5-4488-9bf0-646647e2ca5a.png)
### X_train_grad value
![ML55](https://user-images.githubusercontent.com/118707332/235416703-99e3cea7-46ff-4dac-b5c1-a4a4d3227aee.png)
### Y_train_grad value
![ML56](https://user-images.githubusercontent.com/118707332/235416718-2c9dedfc-a3aa-4773-a007-362a46671799.png)
### Print(res.x)
![ML57](https://user-images.githubusercontent.com/118707332/235416726-9fdb16fa-5ddf-4751-97ae-c35ad0fe5e45.png)
### Decision Boundary graph for Exam Score
![ML58](https://user-images.githubusercontent.com/118707332/235416733-e00d0f7d-737c-4d5a-bb76-543b459ddd69.png)
### Probability value
![ML59](https://user-images.githubusercontent.com/118707332/235416740-090c4473-bc4a-4aeb-bbec-f820749fec95.png)
### Prediction value of mean
![ML510](https://user-images.githubusercontent.com/118707332/235416754-0e976ea9-be9f-4582-9122-5c4db9835a40.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

