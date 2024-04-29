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

dataset['Pos_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['positive'])
dataset['Neg_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['negative'])
dataset['Ang_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['anger'])
dataset['Anti_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['anticipation'])
dataset['Dis_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['disgust'])
dataset['Fear_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['fear'])
dataset['Joy_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['joy'])
dataset['Sad_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['sadness'])
dataset['Surp_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['surprise'])
dataset['Tru_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['trust'])

dataset['Sent_Dic_pos_surp'] = (
            dataset['Pos_Dic'] + dataset['Anti_Dic'] + dataset['Joy_Dic'] + dataset['Surp_Dic'] + dataset['Tru_Dic']
            - dataset['Neg_Dic'] - dataset['Ang_Dic'] - dataset['Dis_Dic'] - dataset['Fear_Dic'] - dataset['Sad_Dic'])

dataset['Sent_Dic_neg_surp'] = (
            dataset['Pos_Dic'] + dataset['Anti_Dic'] + dataset['Joy_Dic'] + dataset['Tru_Dic'] - dataset['Surp_Dic']
            - dataset['Neg_Dic'] - dataset['Ang_Dic'] - dataset['Dis_Dic'] - dataset['Fear_Dic'] - dataset['Sad_Dic'])

dataset.to_csv('../document/AnnualReports16_nrc.csv', index=False)
