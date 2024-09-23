import numpy as np
import pickle
from scipy import signal
from scipy.stats import multivariate_normal
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from einops import rearrange, repeat
# rng = np.random.default_rng()


# get_single_subject_data(subj_id): (40, 60, 32, 5, 128), (40, 3, 32, 5, 128), (40,)
def get_single_subject_data(subj_id=1, remove_ambigous=False):
     with open(f'data/DEAP/data_preprocessed_python/s{str(subj_id).zfill(2)}.dat', 'rb') as file:
          c = pickle.load(file, encoding='latin1')
          data = c['data']
          labels = c['labels']  # (valence, arousal, dominance, liking)

     # only use eeg channels
     data = data[:, :32, :]

     # window BEFORE decomposing bands to prevent data leakage
     data = rearrange(
          data, 'trials channels (chunks window) -> trials chunks channels window', window=128)

     # filter data into bands
     bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
     filters = [signal.butter(3, band, btype='bandpass',
                              fs=128, output='sos') for band in bands]

     data = np.array(list(map(lambda x: signal.sosfilt(x, data, axis=-1), filters)) + [data])

     data = rearrange(data, 'bands trials chunks channels window -> trials chunks channels bands window')

     # split off baselines
     data_baseline = data[:, :3, ...]
     data = data[:, 3:, ...]

     # get labels
     if remove_ambigous:
          admissible_valence_ids = (labels[:, 0] > 5.5) | (labels[:, 0] < 4.5)
          data = data[admissible_valence_ids]
          labels_valence = (labels[admissible_valence_ids, 0] > 5.5).astype(int)
     else:
          labels_valence = (labels[:, 0] > 5).astype(int)  # 1 = HV, 0 = LV

     # (40, 60, 32, 5, 128), (40, 3, 32, 5, 128), (40,)
     return data, data_baseline, labels_valence

# get_single_trial_data(subj_id): (32, 60, 32, 5, 128), (32, 3, 32, 5, 128), (32,)
def get_single_trial_data(trial_id=0):
     data, labels = [], []
     for subj_id in range(1, 33):
          with open(f'data/DEAP/data_preprocessed_python/s{str(subj_id).zfill(2)}.dat', 'rb') as file:
               c = pickle.load(file, encoding='latin1')
               data.append(c['data'][trial_id, :32, ...])
               labels.append(c['labels'][trial_id,0])  # (valence, arousal, dominance, liking)

     data, labels = np.array(data), np.array(labels)

     # window BEFORE decomposing bands to prevent data leakage
     data = rearrange(
          data, 'subjects channels (chunks window) -> subjects chunks channels window', window=128)

     # filter data into bands
     bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
     filters = [signal.butter(3, band, btype='bandpass',
                              fs=128, output='sos') for band in bands]

     data = np.array(list(map(lambda x: signal.sosfilt(x, data, axis=-1), filters)) + [data])

     data = rearrange(data, 'bands trials chunks channels window -> trials chunks channels bands window')

     # split off baselines
     data_baseline = data[:, :3, ...]
     data = data[:, 3:, ...]

     # get labels
     labels_valence = (labels > 5).astype(int)  # 1 = HV, 0 = LV

     # (40, 60, 32, 5, 128), (40, 3, 32, 5, 128), (40,)
     return data, data_baseline, labels_valence

# Single Subject eval
def fair_cross_val(classifier, features, labels, folds=5):
     # expects features of shape (trials, chunks = 60, features)
     # expects labels of shape (trials,)
     n_t = features.shape[0]
     n_c = features.shape[1]
     assert n_t % folds == 0

     folds = np.random.permutation(np.arange(n_t)).reshape((folds, n_t//folds))

     scores = []
     for fold in folds:
          # Get train and test splits according to fold
          # and create feature vectors.
          test_inds = fold
          train_inds = np.setdiff1d(np.arange(n_t), test_inds)
          X_train, X_test = features[train_inds], features[test_inds]
          y_train, y_test = labels[train_inds], labels[test_inds]
          X_train = rearrange(X_train, 'trials chunks ... -> (trials chunks) ...')
          X_test = rearrange(X_test, 'trials chunks ... -> (trials chunks) ...')
          y_train = repeat(y_train, 'lbl -> (lbl chunks)', chunks=n_c)
          y_test = repeat(y_test, 'lbl -> (lbl chunks)', chunks=n_c)

          clf = sklearn.base.clone(classifier)
          clf.fit(X_train, y_train)
          scores.append(clf.score(X_test, y_test))

     return np.array(scores), np.mean(scores), np.var(scores)


class MultivariateGaussModel(ClassifierMixin, BaseEstimator):

     def __init__(self, cov=None):

          self.mus = None
          if cov is not None:
               self.cov = cov
               self.fix_cov = True
          else:
               self.cov = None
               self.fix_cov = False

     def fit(self, X, y=None):
          # We expect X of shape (samples features)
          # We assume that class labels are 0, ..., n
          classes = [X[y == class_name] for class_name in range(max(y)+1)]
          # We assume that covariance within trials is the same across trials
          self.mus = np.array([np.mean(c, axis=0) for c in classes])
          if not self.fix_cov:
               self.cov = np.cov(X, rowvar=False, ddof=1)
          # self.covs = np.array([np.cov(c, rowvar=False) for c in classes])

          return self

     def predict(self, X):

          class_probs = np.stack(
                    [multivariate_normal.pdf(X, mean=mu, cov=self.cov) for mu in self.mus],
                    axis=-1)
          preds = np.argmax(class_probs, axis=-1)
          return preds
