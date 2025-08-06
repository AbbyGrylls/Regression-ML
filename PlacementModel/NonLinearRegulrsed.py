# Non linear polynomial with regularisation model
# i.e f(x) = w1x + w2x^2 + w3x^3 + . . . + wnx^n and lamda- regularisation parameter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('placement.csv', low_memory=False)
varX=df['cgpa'].to_numpy()
varY=df['package'].to_numpy()
m=varX.shape[0]
#cost function
