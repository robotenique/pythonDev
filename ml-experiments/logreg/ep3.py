import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(12)


def logistic_fit(X, y, w=None, batch_size=None, learning_rate=1e-2,
                 num_iterations=1000, return_history=False):
    X_old = X.copy()
    y_old = y.copy()

    # Reshape the data
    X = np.column_stack((np.ones(len(X)), X))
    y = np.clip(y, a_min=0, a_max=None)
    w = np.random.random(X.shape[1]) if w is None else w
    eps = 1e-5
    #decision_boundary(_old, y_old, w.ravel(), save=True, save_id=12)
    w, y, m = w[:, None], y[:, None], X.shape[0]

    hist = np.zeros(num_iterations + 1)
    # Lambda functions for the optimization
    idxs_sample = lambda : np.random.permutation(m)[:batch_size]
    sigmoid = lambda z:  (1 + np.exp(-z))**-1
    grad = lambda theta, X_mat, y_mat: (1/m)*X_mat.T@(sigmoid(X_mat@theta) - y_mat)
    loss = lambda theta, curr_p: (1/m)*(-1*y.T@np.log(curr_p + eps) - (1 - y).T@np.log(1 - curr_p + eps))

    # Gradient descent
    for i in range(num_iterations):
        if return_history:
            hist[i] = loss(w, sigmoid(X@w)).ravel()[0]
        if batch_size is not None:
            batch_idx = idxs_sample()
            w -= learning_rate*grad(w, X[batch_idx, :], y[batch_idx, :])
        else:
            w -= learning_rate*grad(w, X, y)
        #decision_boundary(X_old, y_old, w.ravel(), save=True, save_id=i)


    if return_history:
        hist[len(hist) - 1] = loss(w, sigmoid(X@w))
        return w.ravel(), hist

    return w.ravel()


def predict(X, w):
    sigmoid = lambda z:  (1 + np.exp(-z))**-1
    w = w[:, None]
    X = np.column_stack((np.ones(len(X)), X))
    return sigmoid(X@w).ravel()



# Plot a decision boundary
def decision_boundary(X, y, w, save=False, save_id=0):
    def hyp(theta, X, n):
        h = np.ones((X.shape[0], 1))
        theta = theta.reshape(1, n+1)
        for i in range(0, X.shape[0]):
            h[i] = 1 / (1 + np.exp(-float(np.matmul(theta, X[i]))))
        h = h.reshape(X.shape[0])
        return h
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    X_t = np.concatenate((np.ones((xx.shape[0]*xx.shape[1], 1)),
                          np.c_[xx.ravel(), yy.ravel()]), axis=1)
    h = hyp(w, X_t, 2)
    h = h.reshape(xx.shape)
    plt.contourf(xx, yy, h)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')
    if save:
        plt.savefig(f"{save_id}coisa.png")
        plt.clf()
    else:
        plt.show()


# Testing and evaluation
def gen_lc_batch(X, y):
    """
    Save the plots of learning curves with different batch_size
    """
    for bs in range(1, int(X.shape[0]/2), 100):
        num_iterations = 700
        _, h = logistic_fit(X, y, num_iterations=700, batch_size=bs, return_history=True)
        plt.ylim((0, 5))
        plt.title(f"batch_size = {bs}")
        plt.plot(np.arange(num_iterations + 1), h)
        print(f"Saving {bs}...")
        plt.savefig(f"{bs}lc.png")
        plt.clf()





if __name__ == "__main__":
    """X = np.array([[2, 2], [3, 2]])
    y = np.array([-1, 1])
    w = logistic_fit(X, y)"""
    num = 2500
    x1 = np.concatenate([np.random.normal(0, 4, num),
                         np.random.normal(5, 2, num)
                         ])
    x2 = np.concatenate([np.random.normal(10, 2, num),
                         np.random.normal(10, 2, num)
                         ])
    hue = np.concatenate([np.zeros(num),
                          np.ones(num)
                          ])
    df = pd.DataFrame({"x1": x1,
                       "x2": x2,
                       "hue": hue})

    X = np.concatenate([x1[:, None], x2[:, None]], axis=1)
    y = hue
    w = logistic_fit(X, y, batch_size=int(X.shape[0]/3))
    pred_proba = predict(X, w)
    threshold = .5
    def rpred(y, threshold):
        return 0 if y <= threshold else 1
    rpred_f = np.vectorize(rpred)
    pred = rpred_f(pred_proba, threshold)
    decision_boundary(X, y, w)
    print("Classifier accuracy:" , np.mean(pred == y))
    sns.scatterplot(x="x1", y="x2", hue="hue", style="pred", palette="Set2",
                    data=df.assign(pred=pred), s=50)
