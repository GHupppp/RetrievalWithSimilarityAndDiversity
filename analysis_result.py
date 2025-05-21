import pandas as pd
import matplotlib.pyplot as plt

def file_to_data(dataset_name, k):
    df = {}
    df_summary = pd.DataFrame({'Avg_cos_sim': [], 'Sum_vector': [], 'Pairwise_cos_sim': []})
    df_win_rate = pd.DataFrame({'win_rate_A': [], 'win_rate_S': [], 'win_rate_D': []})
    for lambda_val in range(10, 100, 10):
        name = 'MMR' + str(lambda_val)
        df[name] = pd.read_csv(name + '.csv', usecols=['Avg_cos_sim', 'Sum_vector', 'Pairwise_cos_sim'])
        row_to_add = df[name].mean(axis=0)
        df_summary.loc[len(df_summary)] = row_to_add
    df['SDR'] = pd.read_csv('SDR.csv', usecols=['Avg_cos_sim', 'Sum_vector', 'Pairwise_cos_sim'])
    row_to_add = df['SDR'].mean(axis=0)
    df_summary.loc[len(df_summary)] = row_to_add
    for lambda_val in range(10, 100, 10):
        name = 'MMR' + str(lambda_val)
        df[name]['Avg_cos_sim'] = df[name]['Avg_cos_sim'] - df['SDR']['Avg_cos_sim']
        win_num = (df[name]['Avg_cos_sim'] < 0).sum()
        total_num = df[name]['Avg_cos_sim'].count()
        win_rate_A = win_num / total_num
        df[name]['Sum_vector'] = df[name]['Sum_vector'] - df['SDR']['Sum_vector']
        win_num = (df[name]['Sum_vector'] < 0).sum()
        total_num = df[name]['Sum_vector'].count()
        win_rate_S = win_num / total_num
        df[name]['Pairwise_cos_sim'] = df[name]['Pairwise_cos_sim'] - df['SDR']['Pairwise_cos_sim']
        win_num = (df[name]['Pairwise_cos_sim'] > 0).sum()
        total_num = df[name]['Pairwise_cos_sim'].count()
        win_rate_D = win_num / total_num
        new_row = {'win_rate_A': win_rate_A, 'win_rate_S': win_rate_S, 'win_rate_D': win_rate_D}
        df_win_rate.loc[len(df_win_rate)] = new_row
    df_merge = pd.concat([df_summary, df_win_rate], axis=1)
    file_name = dataset_name + 'k=' + str(k) + '.csv'
    df_merge.to_csv(file_name, encoding='utf-8', index=False)

def file_to_figure(f1, f2, f3, f4):
    col = 7
    df1 = pd.read_csv(f1, usecols=[col])
    df2 = pd.read_csv(f2, usecols=[col])
    df3 = pd.read_csv(f3, usecols=[col])
    df4 = pd.read_csv(f4, usecols=[col])
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.columns = ['V1', 'V2', 'V3', 'V4']
    df['V41'] = df['V4'] - df['V1']
    df['V42'] = df['V4'] - df['V2']
    df['V43'] = df['V4'] - df['V3']
    print(df.describe())
    print(df.mean(axis=0))
    V41 = 0
    V42 = 0
    V43 = 0
    for (index, row) in df.iterrows():
        if row['V41'] > 0:
            V41 += 1
        if row['V42'] > 0:
            V42 += 1
        if row['V43'] > 0:
            V43 += 1
    print('VRSD对MMR（λ=0，0.5，1.0）的胜率：', V41, V42, V43)
    plt.xlim(0, 100)
    plt.ylim(0.3, 1.0)
    plt.plot(df['V1'], color='gray')
    plt.plot(df['V2'], color='green')
    plt.plot(df['V3'], color='blue')
    plt.plot(df['V4'], color='red')
    plt.show()
file_to_data('OpenBook', 15)