import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import brier_score_loss
from scipy.special import expit

from bigraph.datasets import load_wn11
from bigraph.latent_features.models import TransE
import tensorflow as tf
X = load_wn11()
X_valid_pos = X['valid'][X['valid_labels']]
X_valid_neg = X['valid'][~X['valid_labels']]
tf.config.experimental_run_functions_eagerly(False)
model = TransE(batches_count=64, seed=0, epochs=1, k=100, eta=20,
               optimizer='adam', optimizer_params={'lr':0.0001},
               loss='pairwise', verbose=True, large_graphs=False)



model.fit(X['train'])

# Raw scores
scores = model.predict(X['test'])

# Calibrate with positives and negatives
model.calibrate(X_valid_pos, X_valid_neg, positive_base_rate=None)
probas_pos_neg = model.predict_proba(X['test'])
print(probas_pos_neg)
# Calibrate with just positives and base rate of 50%
model.calibrate(X_valid_pos, positive_base_rate=0.5)
probas_pos = model.predict_proba(X['test'])
print(probas_pos)

# Calibration evaluation with the Brier score loss (the smaller, the better)
print("Brier scores")
print("Raw scores:", brier_score_loss(X['test_labels'], expit(scores)))
print("Positive and negative calibration:", brier_score_loss(X['test_labels'], probas_pos_neg))
print("Positive only calibration:", brier_score_loss(X['test_labels'], probas_pos))