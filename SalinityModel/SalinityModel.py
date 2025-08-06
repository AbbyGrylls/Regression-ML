import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('bottle.csv', low_memory=False)
Temperature = df['T_degC'].to_numpy()[:1215]
Depth = df['Depthm'].to_numpy()[:1215]
Oxygen = df['O2ml_L'].to_numpy()[:1215]
Density = df['STheta'].to_numpy()[:1215]
Salinity = df['Salnty'].to_numpy()[:1215]

print("Temperature:", Temperature[:10])
print("Depth:", Depth[:10])
print("Oxygen:", Oxygen[:10])
print("Density:", Density[:10])
print("Salinity:", Salinity[:10])
#varX=[Temperature, Depth, Oxygen, Density]-> its wrong as it creates matrix of 4 rows m columns
varX=np.column_stack([Temperature, Depth, Density]) # didn't add oxygen because
means = varX.mean(axis=0)
stds = varX.std(axis=0)
varX = (varX - means) / stds
varY=Salinity

#computing cost first:
def compute_cost(varX,varY,w,b,m):
    cost=0
    for i in range(m):
        f_wb_i= np.dot(varX[i], w)+b # w and varX[i] are 4x1 and 1x4(4 diff features crosses 4 diff weights) so following matrix rules, dot product acheivable
        cost = cost + (f_wb_i - varY[i])**2
    cost = cost/(2*m)
    return cost
#gradient function:
def compute_grad(varX,varY,w,b,m,n):
    dw=np.zeros((n,)) # n is number of features and m is number of training examples
    db=0
    for i in range(m):
        f_wb_i=np.dot(varX[i], w)+b
        err=f_wb_i-varY[i]
        for j in range(n):
            dw[j]=dw[j]+err*varX[i][j]
        db= db+err
    dw=dw/m
    db=db/m
    return dw,db
#gradient descent:
def grad_des(varX,varY,w,b,iter=100,alpha=0.01):
    m=varX.shape[0]
    n=varX.shape[1]
    J_history = []
    p_history = []
    for i in range(iter):
        dw,db = compute_grad(varX,varY,w,b,m,n)
        w = w - alpha * dw
        b = b - alpha * db
        J_history.append(compute_cost(varX, varY, w, b, m))
        p_history.append((w.copy(), b))
        # Print every 10 steps
        if i % 10 == 0 or i == iter - 1:
            print(f"Iteration {i:4}: Cost {J_history[-1]:.4f}, "
                  f"dj_dw: {dw}, dj_db: {db:.4f}, "
                  f"w: {w}, b: {b:.4f}")
    return w, b, J_history, p_history
w = np.zeros(varX.shape[1])
b = 0
w,b,J_history,p_history=grad_des(varX, varY,w,b)

# Plotting iterations vs w (for each feature) and b
w_history = [p[0] for p in p_history]
b_history = [p[1] for p in p_history]
w_history = np.array(w_history)

plt.figure()
for i in range(w_history.shape[1]):
    plt.plot(w_history[:, i], label=f'w[{i}]')
plt.plot(b_history, label='b')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Weights and b vs. Iterations')
plt.legend()
plt.show()

# 3D plot: w[0], w[1], cost function over iterations (example for first two weights)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(w_history[:, 0], w_history[:, 1], J_history, marker='o', color='blue', label='Gradient Descent Path')
ax.set_xlabel('w[0]')
ax.set_ylabel('w[1]')
ax.set_zlabel('Cost (J)')
ax.set_title('3D Plot of w[0], w[1], and Cost Function')
ax.legend()
plt.show()


# Points learned: (make a report of this)
# no data ref inplace of 0 from csv import cause NaN error
# decresing alpha, quickly converges J
# importance of feature scaling: depth value is too high comparitively in same examples causing cost to diverge
# Improvizations: use different polynomial/nonlinear functions,Or feature engineering may help to improve cost from 500