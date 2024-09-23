import os
import re
from copy import copy
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import model_selection

from torcheeg.datasets.module.base_dataset import BaseDataset
from itertools import product


class trialwiseKFold:
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False):

        self.n_splits = n_splits
        self.shuffle = shuffle

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=None)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        pass

    @property
    def fold_ids(self):
        return list(range(5))

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        info = dataset.info
        trial_ids = list(set(info['trial_id'])) # = range(40)
        subject_ids = list(set(info['subject_id'])) # = list of 32 subjects
        t_x_s = list(product(trial_ids, subject_ids))

        for fold_id, (train_index_trial_ids, test_index_trial_ids) in enumerate(
                self.k_fold.split(t_x_s)):
            #for every fold, split the the (subject x trial) grid into train and test completely randomly
            train_trial_ids = np.array(t_x_s)[train_index_trial_ids].tolist()
            test_trial_ids = np.array(t_x_s)[test_index_trial_ids].tolist()

            train_info = []
            for (train_trial_id, train_subject_id) in train_trial_ids:
                train_info.append(info[(info['trial_id'] == int(train_trial_id)) & (info['subject_id'] == train_subject_id)])
                assert len(train_info[-1]) == 60
            train_info = pd.concat(train_info, ignore_index=True)

            test_info = []
            for (test_trial_id, test_subject_id) in test_trial_ids:
                test_info.append(info[(info['trial_id'] == int(test_trial_id)) & (info['subject_id'] == test_subject_id)])
                assert len(test_info[-1]) == 60
            test_info = pd.concat(test_info, ignore_index=True)

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': None,
            'split_path': None,
        }

    def __repr__(self) -> str:
        return 'My KFold :)'