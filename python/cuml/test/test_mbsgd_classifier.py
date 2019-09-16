# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from cuml.linear_model import MBSGDClassifier as cumlMBSGClassifier
from sklearn.linear_model import SGDClassifier
import pytest
from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics import accuracy_score


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('lrate', ['constant', 'invscaling', 'adaptive'])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
@pytest.mark.parametrize('loss', ['hinge', 'log', 'squared_loss'])
@pytest.mark.parametrize('nrows', [20])
@pytest.mark.parametrize('ncols', [6])
def test_mbsgd_classifier(datatype, lrate, input_type, penalty,
                          loss, nrows, ncols):

    train_rows = int(nrows*0.8)
    X, y = make_classification(n_samples=nrows,
                               n_features=ncols, random_state=0)
    X_test = np.array(X[train_rows:, :], dtype=datatype)
    X_train = np.array(X[:train_rows, :], dtype=datatype)
    y_train = np.array(y[:train_rows, ], dtype=datatype)
    y_test = np.array(y[train_rows:, ], dtype=datatype)

    cu_mbsgd_classifier = cumlMBSGClassifier(learning_rate=lrate, eta0=0.005,
                                             epochs=100, fit_intercept=True,
                                             batch_size=2, tol=0.0,
                                             penalty=penalty)

    cu_mbsgd_classifier.fit(X_train, y_train)
    cu_pred = cu_mbsgd_classifier.predict(X_test).to_array()

    skl_sgd_classifier = SGDClassifier(learning_rate=lrate, eta0=0.005,
                                       max_iter=100, fit_intercept=True,
                                       tol=0.0, penalty=penalty,
                                       random_state=0)

    skl_sgd_classifier.fit(X_train, y_train)
    skl_pred = skl_sgd_classifier.predict(X_test)

    cu_error = accuracy_score(cu_pred, y_test)
    skl_error = accuracy_score(skl_pred, y_test)
    assert(cu_error - skl_error <= 0.02)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('nrows', [20])
@pytest.mark.parametrize('ncols', [6])
def test_mbsgd_classifier_default(datatype,
                                  nrows, ncols):

    train_rows = int(nrows*0.8)
    X, y = make_classification(n_samples=nrows,
                               n_features=ncols, random_state=0)
    X_test = np.array(X[train_rows:, :], dtype=datatype)
    X_train = np.array(X[:train_rows, :], dtype=datatype)
    y_train = np.array(y[:train_rows, ], dtype=datatype)
    y_test = np.array(y[train_rows:, ], dtype=datatype)

    cu_mbsgd_classifier = cumlMBSGClassifier()

    cu_mbsgd_classifier.fit(X_train, y_train)
    cu_pred = cu_mbsgd_classifier.predict(X_test).to_array()

    skl_sgd_classifier = SGDClassifier()

    skl_sgd_classifier.fit(X_train, y_train)
    skl_pred = skl_sgd_classifier.predict(X_test)

    cu_error = accuracy_score(cu_pred, y_test)
    skl_error = accuracy_score(skl_pred, y_test)
    assert(cu_error - skl_error <= 0.02)
