import numpy as np
import pandas as pd
import warnings
# from IPython.display import display
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from sklearn.preprocessing import StandardScaler
import itertools
from tqdm.auto import tqdm
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import rankdata
import pickle

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
models = []
for i in range(4):
    with open('./model/model'+str(i)+'.pickle', 'rb') as f:
        models.append(pickle.load(f))

test = pd.read_csv('./tabular-playground-series-aug-2022/test.csv')
submission = pd.read_csv(
    './tabular-playground-series-aug-2022/sample_submission.csv')

data = test.copy()
data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
data['loading'] = np.log1p(data['loading'])

feature = [f for f in test.columns if f.startswith(
    'measurement') or f == 'loading']

fill_dict = {
    'A': ['measurement_5', 'measurement_6', 'measurement_8'],
    'B': ['measurement_4', 'measurement_5', 'measurement_7'],
    'C': ['measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
    'D': ['measurement_5', 'measurement_6', 'measurement_7', 'measurement_8'],
    'E': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_8'],
    'F': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_7'],
    'G': ['measurement_4', 'measurement_6', 'measurement_8', 'measurement_9'],
    'H': ['measurement_4', 'measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
    'I': ['measurement_3', 'measurement_7', 'measurement_8']
}


for code in data.product_code.unique():
    tmp = data[data.product_code == code]
    column = fill_dict[code]
    tmp_train = tmp[column+['measurement_17']].dropna(how='any')
    tmp_test = tmp[(tmp[column].isnull().sum(axis=1) == 0)
                   & (tmp['measurement_17'].isnull())]
    print(f"code {code} has {len(tmp_test)} samples to fill nan")
    model = HuberRegressor()
    model.fit(tmp_train[column], tmp_train['measurement_17'])
    data.loc[(data.product_code == code) & (data[column].isnull().sum(axis=1) == 0) & (
        data['measurement_17'].isnull()), 'measurement_17'] = model.predict(tmp_test[column])

    model2 = KNNImputer(n_neighbors=5)
    print(f"KNN imputing code {code}")
    data.loc[data.product_code == code, feature] = model2.fit_transform(
        data.loc[data.product_code == code, feature])


def _scale(test_data, feats, i):
    with open('./scalermodel/scalermodel'+str(i)+'.pickle', 'rb') as f:
        scaler = pickle.load(f)
    # scaler = PowerTransformer()

    scaled_test = scaler.transform(test_data[feats])

    new_test = test_data.copy()

    new_test[feats] = scaled_test

    assert len(test_data) == len(new_test)

    return new_test


def model_predict(select_feature, i):
    lr_test = np.zeros(len(test))
    #kf = GroupKFold(n_splits=5)
    x_test = data.copy()

    x_test = _scale(x_test, select_feature, i)
    #x_train, x_val, x_test = x_train.round(2), x_val.round(2), x_test.round(2)

    model = models[i]
    lr_test += model.predict_proba(x_test[select_feature])[:, 1]
    return lr_test


select_feature = ['m3_missing', 'm5_missing', 'measurement_1',
                  'measurement_2', 'loading', 'measurement_17']
submission['lr0'] = model_predict(select_feature, 0)


select_feature = ['measurement_1',
                  'measurement_2', 'loading', 'measurement_17']
submission['lr1'] = model_predict(select_feature, 1)

select_feature = ['m3_missing', 'm5_missing',
                  'measurement_2', 'loading', 'measurement_17']
submission['lr2'] = model_predict(select_feature, 2)

submission.head()

select_feature = ['measurement_2', 'loading', 'measurement_17']
submission['lr3'] = model_predict(select_feature, 3)

submission.head()

submission['rank0'] = rankdata(submission['lr0'])
submission['rank1'] = rankdata(submission['lr1'])
submission['rank2'] = rankdata(submission['lr2'])
submission['rank3'] = rankdata(submission['lr3'])

submission['failure'] = submission['rank0']*0.2 + submission['rank1'] * \
    0.25 + submission['rank2']*0.25 + submission['rank3']*0.3

submission.head()

submission[['id', 'failure']].to_csv('0816023.csv', index=False)
