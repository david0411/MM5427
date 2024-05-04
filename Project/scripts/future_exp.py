import pandas as pd


def sentiment_score(text, sen_list):
    temp_list = []
    for t in text:
        if isinstance(t, str):
            temp = 0
            for w in sen_list:
                temp += t.count(w)
            if len(t) != 0:
                temp_list.append(temp / len(t))
            else:
                temp_list.append(0)
        else:
            temp_list.append(0)
    return temp_list


df = pd.read_csv('../document/Preprocessed_18.csv')

nrc = pd.read_csv('../word_list/NRC-Emotion-Lexicon.txt', sep='\t', names=['term', 'category', 'associated'])
category_list = nrc['category'].unique().tolist()
filtered_df = nrc[nrc['associated'] == 1]
grouped_df = filtered_df.groupby('category')['term'].apply(list)

anti_list = grouped_df.loc['anticipation']

mcd = pd.read_csv('../word_list/Loughran-McDonald_MasterDictionary_1993-2023.csv')
mcd['Word'] = mcd['Word'].str.lower()

neg_list = set(mcd[mcd['Negative'] != 0]['Word'])
pos_list = set(mcd[mcd['Positive'] != 0]['Word'])
unc_list = set(mcd[mcd['Uncertainty'] != 0]['Word'])
lit_list = set(mcd[mcd['Litigious'] != 0]['Word'])
stg_list = set(mcd[mcd['Strong_Modal'] != 0]['Word'])
weak_list = set(mcd[mcd['Weak_Modal'] != 0]['Word'])
ctr_list = set(mcd[mcd['Constraining'] != 0]['Word'])
Comp_list = set(mcd[mcd['Complexity'] != 0]['Word'])

sen_df = pd.DataFrame(df['item7']).copy()
sen_df['Pos_Dic'] = sentiment_score(df['item7'], pos_list)
sen_df['Neg_Dic'] = sentiment_score(df['item7'], neg_list)
sen_df['Anti_Dic'] = sentiment_score(df['item7'], anti_list)

A_P_list = list(set(anti_list) & set(pos_list))
A_N_list = list(set(anti_list) & set(neg_list))
Neutral_list = [x for x in anti_list if x not in A_P_list and x not in A_N_list]
sen_df['pos_anti_score'] = sentiment_score(df['item7'], A_P_list)
sen_df['neg_anti_score'] = sentiment_score(df['item7'], A_N_list)
sen_df['neutral_anti_score'] = sentiment_score(df['item7'], Neutral_list)
sen_df['final_score'] = sen_df['neutral_anti_score'] + sen_df['pos_anti_score'] - sen_df['neg_anti_score']

df2 = pd.DataFrame(sen_df['final_score']).copy()
df2['unc_Dic'] = sentiment_score(df['item7'], unc_list)
df2['stg_Dic'] = sentiment_score(df['item7'], stg_list)
df2['weak_Dic'] = sentiment_score(df['item7'], weak_list)

df2['lit_Dic'] = sentiment_score(df['item7'], lit_list)
df2['ctr_Dic'] = sentiment_score(df['item7'], ctr_list)

df2['unc_risk'] = df2['unc_Dic'] + df2['weak_Dic'] - df2['stg_Dic']
df2['lit_risk'] = df2['lit_Dic'] + df2['ctr_Dic']

sen_df.to_csv('../document/18_result_score1.csv', index=False)
df2.to_csv('../document/18_result_score2.csv', index=False)
