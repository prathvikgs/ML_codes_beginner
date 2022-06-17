import numpy as np
import matplotlib.pyplot as plt

# gradient descent for 1 parameter

# cost funct
def cost_func(x, y, theta_0, theta_1):
    m = len(x)
    J = 0
    for i in range(m):
        J += (theta_0 + x[i] * theta_1 - y[i]) ** 2
    return (J)


# derivative terms
def derivatives(x, y, theta_0, theta_1):
    m = len(x)
    J = 0
    D = [0, 0]

    for i in range(m):
        D[0] += (theta_0 + x[i] * theta_1 - y[i]) / m
        D[1] += (theta_0 + x[i] * theta_1 - y[i]) * x[i] / m

    return (D)


# main loop
x = [3, 8, 1, 4]
y = [-1, 4, -3, 5] #training sets

m = len(x)  # no of learning examples
alpha = 0.001  # learning rate
theta_0 = 0  # initial value
theta_1 = 0  # initial value
t = []
theta = []
D = derivatives(x, y, theta_0, theta_1)
while D[0] ** 2 > 1e-8 and D[1] ** 2 > 1e-8:#setting automatic convergence test
    D = derivatives(x, y, theta_0, theta_1)
    temp0 = theta_0 - alpha * D[0]
    temp1 = theta_1 - alpha * D[1]
    theta_0 = temp0
    theta_1 = temp1
    t.append(cost_func(x, y, theta_0, theta_1))#storing cost function after each step
    theta.append(theta_0)#storing the new value of theta0
    print(("%12.8f %12.8f") % (theta_0, theta_1), end='\r')#printing parameters after each iteration

print(theta_0, theta_1)
print("complete")

x_1 = np.array(x)
y_1 = np.array(y)

plt.plot(x, y, 'x')  # training set plotted
plt.plot(x_1, x_1 * theta_1 + theta_0)
plt.plot(sum(x) / len(x), sum(y) / len(y), 'ok')  # to find if it's working as it plots the avg point

#normal eqn method

X=np.array([[1,3],[1,8],[1,1],[1,4]])
Y=np.array([[-1],[4],[-3],[5]])

THETA=np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),Y))#values of parameter by analytical method


print(THETA)


# In[ ]:




