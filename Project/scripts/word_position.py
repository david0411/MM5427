import pandas as pd
import nltk
from nltk import WordNetLemmatizer

dataset = pd.read_csv('../document/AnnualReports16_processed2.csv')
nrc = pd.read_csv('../word_list/NRC-Emotion-Lexicon.txt',
                  sep='\t', names=['term', 'category', 'associated'])
category_list = nrc['category'].unique().tolist()
filtered_df = nrc[nrc['associated'] == 1]
grouped_df = filtered_df.groupby('category')['term'].apply(list)

anti_list = grouped_df.loc['anticipation']

position = []
stemmer = WordNetLemmatizer()
for document in dataset['item7']:
    if isinstance(document, str):
        words = nltk.word_tokenize(document.lower())
        count = 0
        occur_index = []
        occur_list = []
        for word in words:
            count += 1
            if len(word.strip()) != 0:
                word = stemmer.lemmatize(word)
                for anti_word in anti_list:
                    if word.count(anti_word) > 0:
                        occur_list.append(word)
                        occur_index.append(count)
        position.append(occur_index)
    else:
        position.append([])
dataset['position'] = position
dataset.to_csv('../document/AnnualReports16_word_position.csv', index=False)
