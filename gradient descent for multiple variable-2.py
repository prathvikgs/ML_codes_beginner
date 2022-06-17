#multivariate batch gradient descent

import numpy as np
import matplotlib.pyplot as plt

#calculating the cost function J

def cost_func(theta,x,y):
    m=len(y)
    J=0
    for i in range(m):
        h=np.dot(theta,np.transpose(x[i]))
        J+=(1/m)*(h-y[i])**2
        
    return(J)

#derivative terms

def derivative(theta,x,y):
    m=len(y) #no of training examples
    n=len(theta) #no of features
    d=np.zeros(n)
    
    for j in range(m):
        for i in range(n):
            u=np.dot(theta,np.transpose(x[j]))
            d[i]+=1/n*((u-y[j])*x[j][i])
    return (d)


#taking the training data
x=np.array([[1,3,1],[1,8,4],[1,1,2],[1,4,0]])
y=np.array([-1,4,-3,5])
theta=np.array([0,0,0])
t=[]

#setting some initial values

m=len(y)                      #no of learning set
alpha=1e-3                    #learning rate
theta_ini=np.array([1,0,0])   #initial values
v=derivative(theta_ini,x,y)
k=abs(v)
n=len(theta_ini)              #no of features
t1=[]
temp=np.zeros(n)              #temporaru array required


#the main loop of gradient descent

for it in range(10000):                             #will perform 10000 iterations, we can also keep some convergence results
    temp=theta_ini-alpha*derivative(theta_ini,x,y)  #storig the value of theta-a*der
    
    theta_ini=temp                                  #updating the value
    
    t.append(cost_func(theta_ini,x,y))
    t1.append(theta_ini[0])                         #storing the theta after every iteration
    print(theta_ini)


#prediction the values
y_out=np.dot(x,np.transpose(theta_ini))

#   Analytical method
THETA=np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.dot(np.transpose(x),y

THETA-theta_ini #gives the accuracy difference bw the Analytical method and gradient descent
