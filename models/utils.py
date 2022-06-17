import numpy as np
import pandas as pd


def show_rank_metrics(df_group: pd.DataFrame, df_proba: pd.DataFrame, df_true: pd.DataFrame, label=''):
    print(f'\n------{label}-------')
    df_metric = pd.concat([df_group, df_proba, df_true], axis=1)
    #df_metric = df_metric.sort_values(by=['group', 'proba', 'correct'], ascending=[True, False, True])
    df_metric = df_metric.sort_values(by=['group', 'proba'], ascending=[True, False])
    print(df_metric.shape)
    positions = []
    zeros = []
    group_size = []
    cur_group = -1
    cur_pos = 1
    cur_prob_zeros = 0
    for row in df_metric.itertuples():
        cur_pos += 1
        if cur_group != row.group:
            cur_group = row.group
            zeros.append(cur_prob_zeros)
            group_size.append(cur_pos)
            cur_prob_zeros = 0
            cur_pos = 1
        if row.proba == 0:
            cur_prob_zeros += 1
        if row.correct == 1:
            positions.append(cur_pos)

    print(f'\nmean = {np.mean(positions)}  median={np.median(positions)}, zerosCountMean={np.mean(zeros)}, groupSizeMean={np.mean(group_size)}\n')

    count = [0] * (max(positions) + 1)
    for p in positions:
        count[p] += 1
    acc = 0
    sum_all = sum(count)
    for i, c in enumerate(count):
        acc += c
        print(f'top{i} = {acc / sum_all}')
        if i > 15:
            print(f'stop {i}')
            break
    # positions.sort()
    # print(f'first 1000: {positions[:1000]}')
    print()

#  example
# show_rank_metrics(
#     df_group=pd.DataFrame(data=train['group'].tolist(), columns=['group']),
#     df_proba=pd.DataFrame(data=model.predict_proba(X_train)[:,1], columns=['proba']),
#     df_true=pd.DataFrame(data=y_train.tolist(), columns=['correct'])
# )
