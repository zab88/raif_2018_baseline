import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


def add_diff_features(some_df):
    ll = ['speed', 'force', 'temperature', 'power', 'chem', 'gap', 'size']
    cols = set(some_df.columns)
    for l in ll:
        for i in range(0, 100):
            c1 = str(l) + str(i)
            c2 = str(l) + str(i-1)
            if c1 not in cols:
                continue
            if c2 not in cols:
                continue
            some_df[c1 + '_' + c2] = some_df[c2] / some_df[c1]

    print('ready diff')
    return some_df

df_x = pd.read_csv('../data/X_train.csv')
df_y = pd.read_csv('../data/y_train.csv', header=None)
df_x_test = pd.read_csv('../data/X_test.csv')

df_x = add_diff_features(df_x)
df_x_test = add_diff_features(df_x_test)

clf = CatBoostRegressor(iterations=125000, random_state=42, loss_function='MAE')

clf.fit(df_x, df_y, verbose=200)

res = pd.DataFrame(columns=['res'])
res['res'] = clf.predict(df_x_test)

res.to_csv('submission.csv', header=None, index=False)
