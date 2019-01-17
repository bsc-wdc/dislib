import argparse
import pandas as pd
from scipy import sparse

from dislib.recommendation import ALS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # data
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv('./data/sample_movielens_ratings.txt',
                     delimiter='::',
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1, random_state=666)
    valid_df = pd.read_csv('./data/test.data', names=cols[:3])

    # just in case there are movies/user without rating
    n_m = max(df.movie_id.nunique(), max(df.movie_id) + 1)
    n_u = max(df.user_id.nunique(), max(df.user_id) + 1)

    idx = int(df.shape[0] * 0.8)

    train_df = df.iloc[:idx]
    test_df = df.iloc[idx:]

    train = sparse.csr_matrix(
        (train_df.rating, (train_df.user_id, train_df.movie_id)),
        shape=(n_u, n_m))
    test = sparse.csr_matrix(
        (test_df.rating, (test_df.user_id, test_df.movie_id)))

    valid = sparse.csr_matrix((valid_df.rating,
                               (valid_df.user_id, valid_df.movie_id)))

    als = ALS(convergence_threshold=0.0001, max_iter=5, seed=666,n_f=2)

    als.fit(train, test)

    print("Prediction user 0:\n%s" % als.predict_user(0))

    # cx = sparse.find(valid)
    # preds = []
    # for i, j, r in zip(cx[0], cx[1], cx[2]):
    #     pred = als.predict_user(i)[j]
    #     preds.append(pred)
    #     print("Rating vs prediction: %.1f - %1.f" % (r, pred))
