import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures

def evaluate(y_test, y_pred):
    print(metrics.mean_squared_error(y_test, y_pred))
    print(metrics.r2_score(y_test, y_pred))
    print(math.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return math.sqrt(metrics.mean_squared_error(y_test, y_pred))


data = np.loadtxt("coisa.txt")
df = pd.read_csv("coisa.txt", header=None, delimiter=" ")
""" df.columns = ['X1', 'X2', 'Y']
df.describe()
df.boxplot()
df.hist()
 """# -> In this case we see a good separation in the data, no need for machine learning
# scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
# plt.show()
x_axis = []
y_axis = []
y2_axis = []
for k in range(1, 13):
    x_axis.append(k)
    pol = PolynomialFeatures(degree=k)
    X, y = pol.fit_transform(df.iloc[:, :2]), df.iloc[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    linreg = LinearRegression()
    learnt = linreg.fit(X_train, y_train)
    y_pred = learnt.predict(X_test)
    y_axis.append(evaluate(y_test, y_pred))
    y2_axis.append(metrics.r2_score(y_test, y_pred))

plt.scatter(x_axis, y_axis, label="RMSE")
plt.scatter(x_axis, y2_axis, label="R2 score")
plt.legend()
plt.show()





