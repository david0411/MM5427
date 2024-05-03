import pandas as pd

df_ti = pd.read_csv('../document/18_tense_index.csv', header=0)
df_tp = pd.read_csv('../document/18_tense_position.csv', header=0)
df_wp = pd.read_csv('../document/18_word_position.csv', header=0)
df_r1 = pd.read_csv('../document/18_result_score1.csv', header=0)
df_r2 = pd.read_csv('../document/18_result_score2.csv', header=0)

df = pd.concat([df_ti,df_tp.iloc[:, 1:],df_wp.iloc[:, 1:],df_r1.iloc[:, 1:],df_r2.iloc[:, 1:]], axis=1)
df.to_csv('../document/18_combine.csv', index=False)
