import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import brier_score_loss
from scipy.special import expit

from bigraph.datasets import load_wn11
from bigraph.models import ComplEx, TransE, ConvE, ConvKB, HolE, DistMult, RandomBaseline
import tensorflow as tf

X = load_wn11()
X_valid_pos = X['valid'][X['valid_labels']]
X_valid_neg = X['valid'][~X['valid_labels']]
tf.config.experimental_run_functions_eagerly(False)

import numpy as np

# model = TransE(batches_count=64, seed=0, epochs=0, k=100, eta=20,
#                optimizer='adam', optimizer_params={'lr': 0.0001},
#                loss='pairwise', verbose=True, large_graphs=True)

# model = ConvE(batches_count=1, seed=22, epochs=5, k=100)

# model = ComplEx(
#     batches_count=2,
#     seed=555,
#     epochs=5,
#     k=20,
#     eta=5,
#     loss='pairwise',
#     loss_params={'margin': 1},
#     regularizer='LP',
#     regularizer_params={'p': 2, 'lambda': 0.1})

# model = ConvKB(batches_count=2, seed=22, epochs=1, k=10, eta=1,
#                embedding_model_params={'num_filters': 32, 'filter_sizes': [1],
#                                        'dropout': 0.1},
#                optimizer='adam', optimizer_params={'lr': 0.001},
#                loss='pairwise', loss_params={}, verbose=True)

# model = HolE(batches_count=1, seed=555, epochs=1, k=10, eta=5,
#              loss='pairwise', loss_params={'margin': 1},
#              regularizer='LP', regularizer_params={'lambda': 0.1})

model = DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
                 loss_params={'margin': 5})
# model = RandomBaseline()
model.fit(X['train'])

# scores
scores = model.predict(X['test'])
print("scores: ", scores)
