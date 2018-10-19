from uuid import uuid4
import numpy as np

class Dataset(object):

    def __init__(self, vectors, labels=None):
        self.vectors = vectors
        self.labels = labels

        idx = [uuid4().int for _ in range(len(vectors.shape[0]))]
        self.ids = np.array(idx)

