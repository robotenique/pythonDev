from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def main():
    # Don't print using scientific notation
    np.set_printoptions(suppress=True)

    # Reading from stdin: This is specific for each problem, in this example we read directly from stdin
    l1 = input().split()
    F = int(l1[0]) # Number of features
    N = int(l1[1]) # Number of samples
    vec = np.array([list(map(float, input().split())) for _ in range(N)]) # Dim =  (F + 1) x N
    instances2predict = int(input()) # Number of instances to predict after training
    to_predict = np.array([list(map(float, input().split())) for _ in range(instances2predict)]) # Examples to predict

    # Separate X and Y
    X, Y = vec[:, :len(vec[0]) - 1], vec[:, len(vec[0]) - 1]
    #print(f"l1 = {l1}\nF = {F}\nN={N}\nvec={vec}\ninstances2predict={instances2predict}\nto_predict={to_predict}")

    # Preprocessing: I know that the data is already normalized & mapped to (0, 1), so no need to do that
    # Polynomial features: Generate a polynomial feature of degree 3, and transform X
    pol3 = PolynomialFeatures(3)
    X_transf = pol3.fit_transform(X)

    # Regression
    # Using a Linear Regression + Polynomial Features = Polynomial Regression
    regressor = LinearRegression() # Least squares regression
    regressor.fit(X_transf, Y)

    prediction = regressor.predict(pol3.fit_transform(to_predict))
    for p in prediction:
        print(p)


if __name__ == '__main__':
    main()