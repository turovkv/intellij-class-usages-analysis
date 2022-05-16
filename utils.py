import numpy as np
import pandas as pd


def show_rank_metrics(df_group: pd.DataFrame, df_proba: pd.DataFrame, df_true: pd.DataFrame):
    df_metric = pd.concat([df_group, df_proba, df_true], axis=1)
    df_metric = df_metric.sort_values(by=['group', 'proba', 'correct'], ascending=[True, False, True])
    print(df_metric.shape)
    positions = []
    cur_group = -1
    cur_pos = 1
    for row in df_metric.itertuples():
        cur_pos += 1
        if cur_group != row.group:
            cur_group = row.group
            cur_pos = 1
        if row.correct == 1:
            positions.append(cur_pos)

    print(f'\nmean = {np.mean(positions)}\n')

    count = [0] * (max(positions) + 1)
    for p in positions:
        count[p] += 1
    acc = 0
    sum_all = sum(count)
    for i, c in enumerate(count):
        acc += c
        print(f'top{i} = {acc / sum_all}')

#  example
# show_rank_metrics(
#     df_group=pd.DataFrame(data=train['group'].tolist(), columns=['group']),
#     df_proba=pd.DataFrame(data=model.predict_proba(X_train)[:,1], columns=['proba']),
#     df_true=pd.DataFrame(data=y_train.tolist(), columns=['correct'])
# )
