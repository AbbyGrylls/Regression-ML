# Non linear polynomial with regularisation model
# i.e f(x) = w1x + w2x^2 + w3x^3 + . . . + wnx^n and lamda- regularisation parameter
# the above is for uni-variable regression and the following is for multi-variable
# f(x) = (w1₁·x₁ + w1₂·x₂ + w1₃·x₃) + (w2₁·x₁ + w2₂·x₂ + w2₃·x₃)^2+ (w3₁·x₁ + w3₂·x₂ + w3₃·x₃)^3+ b
# for matrix form: f(x)= w1x + w2x^2 + w3x^3. w1,w2,w3 are column vectors and coeff of each x powers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load, Structure and Normalize data
df = pd.read_csv('bottle.csv', low_memory=False)
Temperature = df['T_degC'].to_numpy()[:1215]
Depth = df['Depthm'].to_numpy()[:1215]
Oxygen = df['O2ml_L'].to_numpy()[:1215]
Density = df['STheta'].to_numpy()[:1215]
Salinity = df['Salnty'].to_numpy()[:1215]
varX=np.column_stack([Temperature, Depth, Density]) # didn't add oxygen level because data inavailability
means = varX.mean(axis=0)
stds = varX.std(axis=0)
varX = (varX - means) / stds
varY=Salinity
# computing fucntion value: use the above formula
def compute_func(x, w, n, b):
    """
    x: input feature vector (num_features,)
    w: weight matrix (num_features, n)
    n: highest power (degree)
    b: bias
    """
    result = 0
    for power in range(1,n + 1):
        result += np.dot(x, w[:, power - 1]) ** power
    result += b
    return result  
# computing cost first:
def compute_cost(varX,varY,w1,w2,w3,b,m,lmda,n):
    cost = 0
    # stacking w1, w2, w3 into a weight matrix of shape (num_features, n)
    w = np.column_stack([w1, w2, w3])
    for i in range(m):
        f_wb_i = compute_func(varX[i], w, n, b)
        cost += (f_wb_i - varY[i]) ** 2
    cost = cost / (2 * m)
    # regularization (excluding bias)
    cost += (lmda / (2 * m)) * (np.sum(w1 ** 2) + np.sum(w2 ** 2) + np.sum(w3 ** 2))
    return cost
# computing gradient function:
def compute_grad(varX,varY,w1,w2,w3,b,m,lmda,n):
    """
    Computes gradients for w1, w2, w3, and b for the given cost function.
    Returns: dw1, dw2, dw3, db (all shapes match their respective weights)
    """
    num_features = varX.shape[1]
    dw1 = np.zeros(num_features)
    dw2 = np.zeros(num_features)
    dw3 = np.zeros(num_features)
    db = 0
    # Stack weights for use in compute_func
    w = np.column_stack([w1, w2, w3])
    for i in range(m):
        x = varX[i]
        y = varY[i]
        f_wb_i = compute_func(x, w, n, b)
        err = f_wb_i - y
        dw1 += err * (np.dot(x, w[:, 0]) ** 0) * x  # derivative wrt w1: x
        dw2 += err * 2 * (np.dot(x, w[:, 1])) * x   # derivative wrt w2: 2*(w2·x)*x
        dw3 += err * 3 * (np.dot(x, w[:, 2]) ** 2) * x  # derivative wrt w3: 3*(w3·x)^2*x
        db += err
    dw1 = dw1 / m + (lmda / m) * w1
    dw2 = dw2 / m + (lmda / m) * w2
    dw3 = dw3 / m + (lmda / m) * w3
    db = db / m
    return dw1, dw2, dw3, db
# computing gradient descent:
def gradient_descent(varX, varY, w1, w2, w3, b, m, lmda, n, alpha, num_iters):
    """
    Performs gradient descent to optimize w1, w2, w3, and b.
    Returns: w1, w2, w3, b, cost_history (list)
    """
    cost_history = []
    for i in range(num_iters):
        dw1, dw2, dw3, db = compute_grad(varX, varY, w1, w2, w3, b, m, lmda, n)
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        w3 = w3 - alpha * dw3
        b = b - alpha * db
        if i % 10 == 0 or i == num_iters - 1:
            cost = compute_cost(varX, varY, w1, w2, w3, b, m, lmda, n)
            cost_history.append(cost)
    return w1, w2, w3, b, cost_history

# init parameters
num_features = varX.shape[1]
w1 = np.zeros(num_features)
w2 = np.zeros(num_features)
w3 = np.zeros(num_features)
b = 0
m = varX.shape[0]
n = 3
lmda = 0.01 
alpha = 0.01 
num_iters = 500
w1, w2, w3, b, cost_history = gradient_descent(varX, varY, w1, w2, w3, b, m, lmda, n, alpha, num_iters)

plt.figure(1)
plt.plot(np.arange(0, len(cost_history)*10, 10), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations')

preds = []
w = np.column_stack([w1, w2, w3])
for i in range(m):
    preds.append(compute_func(varX[i], w, n, b))
preds = np.array(preds)

plt.figure(2)
plt.scatter(varY, preds, alpha=0.5)
plt.xlabel('Actual Salinity')
plt.ylabel('Predicted Salinity')
plt.title('Actual vs Predicted Salinity')
plt.plot([varY.min(), varY.max()], [varY.min(), varY.max()], 'r--')
plt.show()