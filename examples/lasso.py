import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def main():
    # #########################################################################
    # Generate some sparse data to play with
    np.random.seed(42)

    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features)

    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = (-1) ** idx * np.exp(-idx / 10)
    coef[10:] = 0  # sparsify coef
    y = np.dot(X, coef)

    # Add noise
    y += 0.01 * np.random.normal(size=n_samples)

    # Split data in train set and test set
    n_samples = X.shape[0]
    X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
    X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

    # #########################################################################
    # Lasso dislib
    from dislib.regression import Lasso
    import dislib as ds

    alpha = 0.1
    lasso = Lasso(lmbd=alpha, max_iter=50)

    lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
    y_pred_lasso = lasso.predict(ds.array(X_test, (25, 100)))
    r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())
    print(lasso)
    print("r^2 on test data : %f" % r2_score_lasso)

    # #########################################################################
    # Lasso sklearn
    from sklearn.linear_model import Lasso

    alpha = 0.1
    lasso_sk = Lasso(alpha=alpha)

    y_pred_lasso_sk = lasso_sk.fit(X_train, y_train).predict(X_test)
    r2_score_lasso_sk = r2_score(y_test, y_pred_lasso_sk)
    print(lasso_sk)
    print("r^2 on test data : %f" % r2_score_lasso_sk)

    # #########################################################################
    # ElasticNet
    from sklearn.linear_model import ElasticNet

    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)

    m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0],
                       markerfmt='x', label='Elastic net coefficients',
                       use_line_collection=True)
    plt.setp([m, s], color="#2ca02c")

    m, s, _ = plt.stem(np.where(lasso_sk.coef_)[0], lasso_sk.coef_[
        lasso_sk.coef_ != 0],
                       markerfmt='x', label='Lasso (SK) coefficients',
                       use_line_collection=True)
    plt.setp([m, s], color='#af1b32')

    lasso_coef = lasso.coef_.collect()

    m, s, _ = plt.stem(np.where(lasso_coef)[0], lasso_coef[lasso_coef != 0],
                       markerfmt='x', label='Lasso (dislib) coefficients',
                       use_line_collection=True)
    plt.setp([m, s], color='#ff7f0e')

    plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',
             markerfmt='bx', use_line_collection=True)

    plt.legend(loc='best')
    plt.title("Lasso (ds) $R^2$: %.3f, Lasso (sk) $R^2$: %.3f, Elastic Net "
              "$R^2$: %.3f" % (
                  r2_score_lasso, r2_score_lasso_sk, r2_score_enet))
    plt.show()


if __name__ == "__main__":
    main()
