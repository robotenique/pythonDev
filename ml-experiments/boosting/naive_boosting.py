import numpy as np
import pandas as pd
import matplotlib
import math
import gc
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# config display
sns.set()
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['image.cmap'] = 'viridis'


def std_agg(cnt, s1, s2):
    return math.sqrt((s2/cnt) - (s1/cnt)**2)

# From fast.ai implementation: https://github.com/fastai/fastai/blob/a913af737fbb2e98f89cbcd5ae0e6a8269777859/courses/ml1/lesson3-rf_foundations.ipynb
class DecisionTree(object):
    def __init__(self, x: pd.DataFrame, y: np.array, idxs=None, min_leaf=2):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        # rows and columns
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        # Find a better split by scanning all features
        for i in range(self.c):
            self.find_better_split(i)
        if self.score == float('inf'):
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]  # data which is less than the split
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n-self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1
            rhs_cnt -= 1
            lhs_sum += yi
            rhs_sum -= yi
            lhs_sum2 += yi**2
            rhs_sum2 -= yi**2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self):
        return self.x.columns[self.var_idx]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        # Vectorial predict
        if isinstance(x, pd.DataFrame):
            return np.array([self.predict_row(row) for _, row in x.iterrows()])
        return np.array([self.predict_row(row) for row in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)

    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s


# ------------------ Sample data ------------------
def generate_data(plot=True):
    x = pd.DataFrame({"x": np.arange(50)})
    # random uniform distributions in different ranges
    y1 = np.random.uniform(10, 15, 10)
    y2 = np.random.uniform(20, 25, 10)
    y3 = np.random.uniform(0, 5, 10)
    y4 = np.random.uniform(30, 32, 10)
    y5 = np.random.uniform(13, 17, 10)
    y = np.concatenate((y1, y2, y3, y4, y5))
    df = x.assign(y=y)
    y = y[:, None]
    if plot:
        # plot simulated data
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.scatterplot(x="x", y="y", data=df, ax=ax)
        ax.set_title("Scatter plot of x vs y", fontsize=20)
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("y", fontsize=15)
        plt.tight_layout()
        plt.show()
    return df[["x"]], y


def generate_data_normal(plot=True):
    x = pd.DataFrame({"x": np.arange(100)})
    # random normal distributions in different ranges
    y1 = np.random.normal(13, 3, size=20)
    y2 = np.random.uniform(23, 2, size=20)
    y3 = np.random.uniform(3, 1, size=20)
    y4 = np.random.uniform(31, .5, size=20)
    y5 = np.random.uniform(15, .3, size=20)
    y = np.concatenate((y1, y2, y3, y4, y5))
    df = x.assign(y=y)
    y = y[:, None]
    if plot:
        # plot simulated data
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.scatterplot(x="x", y="y", data=df, ax=ax)
        ax.set_title("Scatter plot of x vs y", fontsize=20)
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("y", fontsize=15)
        plt.tight_layout()
        plt.show()
    return df[["x"]], y


def generate_data_cos(plot=True):
    x = pd.DataFrame({"x": np.linspace(0, 100, 1000)})
    y = 25*np.cos(x["x"]/7) + np.random.normal(0, 8, size=1000)
    df = x.assign(y=y)
    y = y[:, None]
    if plot:
        # plot simulated data
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.scatterplot(x="x", y="y", data=df, ax=ax)
        ax.set_title("Scatter plot of x vs y", fontsize=20)
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("y", fontsize=15)
        plt.tight_layout()
        plt.show()
    return df[["x"]], y


# ------------------ Simple gradient boosting (using default residuals) ------------------
def run_trees(x, y, show_plt=False, plot_notebook=False, num_estimators=10, modulo_step=1):
    """Create up to 'num_estimators' different trees tree stumps, trained on the traditional residual.
    It also plots the intermediary graphs, with support to plotting in Jupyter notebook.

    Important: This works only when X have only one feature, and Y is a regression target! The x dataframe needs to
    have  a column 'x'


    Parameters:
    -----------
    x: dataframe of the features, have a column 'x'
    y: A single numpy column array of the target
    show_plt: flag for showing the plot in a normal python window (plt.show())
    plot_notebook: flag for plotting in jupyter notebook
    num_estimators: The number of weak learners to generate
    modulo_step: number to step the visualizations display. For example, if modulo_step = 5, will only show
                 plots in multiples of 5. It's useful to speed up the visualization when there are a lot of
                 estimators.
    """

    xi = x
    yi = y
    ei = 0
    n = len(yi)
    predf = 0
    rows = np.arange(num_estimators)
    rows = rows[rows % modulo_step == 0]

    if not plot_notebook:
        fig, axs = plt.subplots(len(rows), 2, sharey=True, figsize=(13, 20))

    for i in range(num_estimators):
        tree = DecisionTree(xi, yi)
        tree.find_better_split(0)
        r = np.where(xi == tree.split)[0][0] # index where best split occurs
        left_idx = np.where(xi <= tree.split)[0]
        right_idx = np.where(xi > tree.split)[0]

        predi = np.zeros(n)
        # left side mean
        np.put(predi, left_idx, np.repeat(np.mean(yi[left_idx]), r))
        # right side mean
        np.put(predi, right_idx, np.repeat(np.mean(yi[right_idx]), n - r))
        predi = predi[:, None]  # Transform a vector into a column matrix
        # Update the prediction
        predf = predf + predi

        # calculate residual
        ei = y - predf
        yi = ei  # New y is the residual
        # Plot current prediction
        order = np.argsort(np.array(x["x"]))
        curr_pred_df = pd.DataFrame({"xaxis": np.array(x)[order].ravel(),
                                     "yaxis": np.array(predf)[order].ravel()})
        if i in rows:
            if not plot_notebook:
                ax1 = axs[i, 0]
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 2.5))
            sns.scatterplot(x="x", y="y", data=x.assign(y=y), ax=ax1)
            sns.lineplot(x="xaxis", y="yaxis", data=curr_pred_df, ax=ax1, color="r")
            ax1.set_title(f"Prediction i = {i + 1}", fontsize=20)
            ax1.set_xlabel("x", fontsize=12)
            ax1.set_ylabel("y and y_pred", fontsize=12)
            ax1.legend(labels=['data points', 'prediction'], loc=4)
            leg = ax1.get_legend()
            leg.legendHandles[0].set_color('b')
            leg.legendHandles[1].set_color('r')
    if show_plt:
        plt.show()


if __name__ == "__main__":
    # Uncomment the examples below to check different functionalities
    #generate_data_cos()
    #run_trees(*generate_data(plot=False), show_plt=True)
    from pprint import pprint
    diabetes = load_diabetes()
    x, y = diabetes["data"], diabetes["target"][:, None]
    x = pd.DataFrame({f: x[:, i] for i, f in enumerate(diabetes["feature_names"])})
    train_idx = int(.7 * len(x))  # 70% train split
    x_train, y_train = x.iloc[:train_idx, :], y[:train_idx, :]
    x_test, y_test = x.iloc[train_idx:, :], y[train_idx:, :]
    print(f"full dataset shape: {x.shape}")
    print(f"Shapes: \nx_train: {x_train.shape}, y_train: {y_train.shape} \nx_test: {x_test.shape}, y_test: {y_test.shape}")
    #tree = DecisionTree(x_train, y_train, min_leaf=5)
    #y_pred = tree.predict(x_test)
    #print(f"mean_squared_error: {mean_squared_error(y_test, y_pred)}")
    mse_eval = []
    for depth in range(1, 20):
        tree_regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree_regressor = tree_regressor.fit(x_train, y_train)
        y_pred = tree_regressor.predict(x_test)    
        print(f"mean_squared_error(sklearn): {mean_squared_error(y_test, y_pred)}")
        mse_eval.append(mean_squared_error(y_test, y_pred))
    plt.plot(range(1, 20), mse_eval, marker="o")
    plt.title("Learning curve")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("max depth")
    plt.xticks(range(1, 20))
    plt.show()