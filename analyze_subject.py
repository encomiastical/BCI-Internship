import numpy as np
from einops import rearrange, repeat
from my_utils import *
import sklearn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC



def predict_trial_labels(dat, classifier = SVC(C=4)):
    # We expect dat of shape (trials chunks channels bands)
    nr_trials = dat.shape[0]
    nr_chunks = dat.shape[1]
    lbl = np.repeat(np.arange(nr_trials), nr_chunks)
    clf = sklearn.base.clone(classifier)
    return cross_val_score(clf, rearrange(dat, 'trials chunks channels bands -> (trials chunks) (channels bands)'), lbl, cv = KFold(shuffle = True))

def predict_valence_labels(dat, lbl, classifier = SVC(C=4)):
    # We expect dat of shape (trials chunks channels bands)
    # and lbl of shape (trials)
    nr_chunks = dat.shape[1]
    dat = rearrange(dat, '... channels bands -> ... (channels bands)')
    clf = sklearn.base.clone(classifier)
    # Treat all the chunks as separate samples without any restrictions
    s1 = cross_val_score(clf, rearrange(dat, 'trials chunks ... -> (trials chunks) ...'), repeat(lbl, 'trials -> (trials chunks)', chunks=nr_chunks), cv = KFold(shuffle = True))
    clf = sklearn.base.clone(classifier)
    # Treat all the chunks as separate samples, but create train/test splits by trial
    s2 = fair_cross_val(clf, dat, lbl, folds=5)
    return {'paradigm 1': s1, 'paradigm 2': s2[0]}

def predict_trial_labels_of_test_set_using_folds(dat, dat_base, classifier = SVC(C=4)):
    # We expect dat of shape (trials chunks channels bands)
    # and dat_base of shape (trials chunks channels bands)
    # where ordering of trials of dat matches ordering of trials of dat_base
    nr_chunks = dat.shape[1]
    nr_chunks_base = dat_base.shape[1]
    nr_trials = dat.shape[0]

    lbl_trials = np.arange(nr_trials)
    dat = rearrange(dat, 'trials chunks channels bands -> (trials chunks) (channels bands)')
    dat_base = rearrange(dat_base, 'trials chunks channels bands -> (trials chunks) (channels bands)')
    lbl = repeat(lbl_trials, 'trials -> (trials chunks)', chunks=nr_chunks)
    lbl_base = repeat(lbl_trials, 'trials -> (trials chunks)', chunks=nr_chunks_base)
    # To have similar circumstances like in real training, we also divide into 5 random folds
    nr_folds = 5
    assert nr_trials % nr_folds == 0
    # We permute indices belonging to one trial
    rng = np.random.default_rng(seed=0)
    folds = rng.permutation(np.arange(dat.shape[0]).reshape((nr_trials,nr_chunks)), axis=1)
    # and create folds by splitting the chunks in 5 parts.
    folds = rearrange(folds, 'trials (folds chunks) -> folds (trials chunks)', folds=nr_folds)

    scores = []
    for fold in folds:
        train_inds = np.setdiff1d(np.arange(dat.shape[0]), fold)
        X_train, y_train = dat[train_inds], lbl[train_inds]

        clf = sklearn.base.clone(classifier)
        clf.fit(X_train, y_train)
        scores.append(clf.score(dat_base, lbl_base))

    return np.array(scores)

def analyze_subject(de_main, de_base, trial_labels):

    results = {}

    #1)
    # To what degree are the trials recoverable from the data?
    results['acc predicting trial labels'] = predict_trial_labels(de_main)

    #2)
    results['acc predicting valence labels'] = predict_valence_labels(de_main, trial_labels)

    #3)
    # Shuffling the labels across trials
    lbl = np.random.permutation(trial_labels)
    results['acc predicting shuffled valence labels'] = predict_valence_labels(de_main, lbl)

    #4)
    nr_chunks = 20
    chunks_base = [1,2]
    dat = de_main[:,:nr_chunks,:,:]
    dat_base = de_base[:,chunks_base,:,:]
    #dat_base = rng.normal(size = dat_base.shape, loc = np.mean(dat), scale = np.std(dat, ddof=1))
    results['acc predicting baseline trial labels'] = predict_trial_labels_of_test_set_using_folds(dat, dat_base)

    return results

def compare_on_fixed_train_test_using_folds(train, test, trial_labels, nr_folds = 5, classifier = SVC(C=4)):
    # We expect input of shape
    # - train, test: (trials, chunks, channels, bands)
    # - trial_labels: (trials, )
    nr_trials, nr_train_chunks = train.shape[:2]
    assert nr_train_chunks % nr_folds == 0
    assert nr_trials % nr_folds == 0

    # Creating feature vectors
    X_train = rearrange(train, 'trials chunks channels bands -> trials chunks (channels bands)')
    y_train = repeat(trial_labels, 'trials -> trials chunks', chunks=nr_train_chunks)
    X_test = rearrange(test, 'trials chunks channels bands -> trials chunks (channels bands)')
    y_test = repeat(trial_labels, 'trials -> trials chunks', chunks=test.shape[1])

    # We first do 2)
    # For evaluation of 2) we need to have trials of the test set which are not present in the train set
    # Therefore we choose to take nr_trials/nr_folds test trials and the rest will be train trials

    # Create test folds of all trials:
    rng = np.random.default_rng(seed = 0)
    folds = rng.permutation(np.arange(nr_trials)).reshape((nr_folds,nr_trials//nr_folds))

    scores_2 = []
    for fold in folds:
        # Create test fold
        train_inds = np.setdiff1d(np.arange(nr_trials), fold)
        X_train_2 = rearrange(X_train[train_inds], 'trials chunks ... -> (trials chunks) ...')
        y_train_2 = rearrange(y_train[train_inds], 'trials chunks ... -> (trials chunks) ...') 
        # Classify
        clf = sklearn.base.clone(classifier)
        clf.fit(X_train_2, y_train_2)
        scores_2.append(clf.score(  rearrange(X_test[fold], 'trials chunks ... -> (trials chunks) ...'),
                                    rearrange(y_test[fold], 'trials chunks ... -> (trials chunks) ...')))

    # Then 1)
    # Here we don't care about trials anymore so we can just collapse
    X_train = rearrange(X_train, 'trials chunks ... -> (trials chunks) ...')
    y_train = rearrange(y_train, 'trials chunks ... -> (trials chunks) ...')
    # We also need to remember the folds we used above, see below
    folds_2 = folds
    # For fairness, we only want as many training samples as we used above for 2)
    # We want also every training fold to contain samples from all trials
    # We further want to also use nr_folds folds approach, meaning we do nr_folds iterations where we exclude
    # respectively different chunks
    folds = rng.permutation(np.arange(X_train.shape[0]).reshape((nr_trials,nr_train_chunks)), axis=1)
    folds = rearrange(folds, 'trials (folds chunks) -> folds (trials chunks)', folds=nr_folds)

    scores_1 = []
    for fold_train, fold_test in zip(folds, folds_2):
        train_inds = np.setdiff1d(np.arange(X_train.shape[0]), fold_train)
        X_train_1, y_train_1 = X_train[train_inds], y_train[train_inds]

        clf = sklearn.base.clone(classifier)
        clf.fit(X_train_1, y_train_1)
        # Now we score over THE SAME baselines we scored over in 2), so that the results are comparable
        scores_1.append(clf.score(  rearrange(X_test[fold_test], 'trials chunks ... -> (trials chunks) ...'),
                                    rearrange(y_test[fold_test], 'trials chunks ... -> (trials chunks) ...')))

    return scores_1, scores_2