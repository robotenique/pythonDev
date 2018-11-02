import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix

def main():
    # Don't print using scientific notation
    matplotlib.style.use('ggplot')
    np.set_printoptions(suppress=True)
    #bt_life = float(input(""))
    df = pd.read_csv("trainingdata.txt", header=None)
    df.columns = ["X", "Y"]
    # Separate X and Y
    # NOTE: reshape(-1, 1) generates a column vector, while reshape(1, -1) a row vector
    X, Y = np.array(df.X).reshape(-1, 1), np.array(df.Y).reshape(-1, 1)
    """This is an example of why it's best to analyze the data before jumping into any ML technique
    """
    # analyze(df)
    # In the end, we can use very simple answers, in this casa
    if bt_life >= 0 and bt_life <= 4:
        return 2*bt_life
    elif bt_life > 4:
        return 8
    else:
        return 0

def analyze(df):
    """ Use some useful methods to analyze the data beforehand
        https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
    """
    # Basic statistics about the data
    print(df.describe()) # -> The max (Y) is 8 hours
    # Show a whisker boxplot
    df.boxplot()
    # Histogram of every column
    df.hist()
    # Scatter matrix using Kernel Density Estimator
    # -> In this case we see a good separation in the data, no need for machine learning
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()





if __name__ == '__main__':
    main()
