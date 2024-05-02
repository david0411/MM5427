import nltk
import pandas as pd


def count_sentences(text):
    if pd.isnull(text):
        return 0
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def delete_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) > 2:
        return ' '.join(sentences[2:])
    else:
        return text


df = pd.read_csv('../document/AnnualReports1618.csv', header=0)
df['sentence_count'] = df['item7'].apply(count_sentences)
df = df[df['sentence_count'] > 10]
df_16 = df[df['filed_date'] < 20170000]
df_17 = df[(df['filed_date'] > 20170000) & df['filed_date'] < 20180000]
df_18 = df[df['filed_date'] > 20180000]

df_16.to_csv('../document/filtered_16.csv', index=False)
df_17.to_csv('../document/filtered_17.csv', index=False)
df_18.to_csv('../document/filtered_18.csv', index=False)
