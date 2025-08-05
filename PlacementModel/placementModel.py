import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('placement.csv', low_memory=False)
#extract variables
varX = df['cgpa'].to_numpy()
varY = df['package'].to_numpy()
m = varX.shape[0]
#visualizing data
""" plt.scatter(varX,varY,color="green",marker="o",alpha=0.5)
plt.xlabel("Visualising Data")
plt.xlabel("cgpa")
plt.ylabel("package")
plt.grid(True)
plt.show() """
#cost function
def compute_cost(varX, varY, w, b, m):
    cost = 0
    for i in range(m):
        f_x = w * varX[i] + b
        cost += (f_x - varY[i])**2
    tcost = cost / (2 * m)
    return tcost
#gradient function
def compute_grad(varX, varY, w, b, m):
    dw = 0
    db = 0
    for i in range(m):
        f_x = w * varX[i] + b
        dw += (f_x - varY[i]) * varX[i]
        db += (f_x - varY[i])
    dw /= m
    db /= m
    return dw, db
#gradient descent
def grad_des(varX, varY, w , b,iterations=100, alpha=0.01):
    m = varX.shape[0]
    J_history = []
    p_history = []
    for i in range(iterations):
        dw, db = compute_grad(varX, varY, w, b, m)
        w = w - alpha * dw
        b = b - alpha * db
        J_history.append(compute_cost(varX, varY, w, b, m))
        p_history.append((w, b))
        # Print every 10 steps
        if i % 10 == 0 or i == iterations - 1:
            print(f"Iteration {i:4}: Cost {J_history[-1]:.4f}, "
                  f"dj_dw: {dw:.4f}, dj_db: {db:.4f}, "
                  f"w: {w:.4f}, b: {b:.4f}")
    
    return w, b, J_history, p_history
w = -1
b = -1
w,b,J_history,p_history=grad_des(varX, varY,w,b)

#plotting the fitted curve: extra can plot iter vs cost function value over iteration. just take J_history plot function
y_pred=w*varX+b
plt.figure(figsize=(8,5))
plt.scatter(varX,varY,color="green",marker="o",alpha=0.5,label="Actual Data")
plt.plot(varX,y_pred,color='red',label=f'prediction line (w={w},b={b})')
plt.title("Linear regression plot")
plt.xlabel("cgpa")
plt.ylabel("predicted package")
plt.legend()
plt.grid(True)
plt.show()

#plotting iterations vs w,b
w_history = [p[0] for p in p_history]
b_history = [p[1] for p in p_history]
plt.plot(w_history, label='w')
plt.plot(b_history, label='b')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('w and b vs. Iterations')
plt.legend()
plt.show()

#3d plot: w,b,j-cost function over iterations
w_history = [p[0] for p in p_history]
b_history = [p[1] for p in p_history]
cost_history = J_history
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(w_history, b_history, cost_history, marker='o', color='blue', label='Gradient Descent Path')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost (J)')
ax.set_title('3D Plot of w, b, and Cost Function')
ax.legend()
plt.show()
#Solution always converges at global minima: because of linear function(convex functions type)