import re
import pandas as pd


def sentiment_score(text, sen_list):
    temp_list = []
    for t in text:
        if isinstance(t, str):
            t = re.sub("[^A-Za-z]+", "", t)
            t = t.lower()
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


dataset = pd.read_csv('../document/AnnualReports16_processed.csv', header=0)
lexicon = pd.read_csv('../word_list/NRC-Emotion-Lexicon.txt', sep='\t', names=['term', 'category', 'associated'])

category_list = lexicon['category'].unique().tolist()
filtered_df = lexicon[lexicon['associated'] == 1]
grouped_df = filtered_df.groupby('category')['term'].apply(list)

dataset['Pos_Anti_Dic'] = sentiment_score(dataset['processed_text'],
                                          set(grouped_df.loc['anticipation']).intersection(grouped_df.loc['positive']))
dataset['Neg_Anti_Dic'] = sentiment_score(dataset['processed_text'],
                                          set(grouped_df.loc['anticipation']).intersection(grouped_df.loc['negative']))

dataset.to_csv('../document/AnnualReports16_nrc_v2.csv', index=False)
