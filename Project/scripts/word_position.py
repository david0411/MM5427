import pandas as pd
import nltk

df = pd.read_csv('../document/Preprocessed_18.csv')
nrc = pd.read_csv('../word_list/NRC-Emotion-Lexicon.txt',
                  sep='\t', names=['term', 'category', 'associated'])

category_list = nrc['category'].unique().tolist()
filtered_df = nrc[nrc['associated'] == 1]
grouped_df = filtered_df.groupby('category')['term'].apply(list)

anti_list = grouped_df.loc['anticipation']

position = []
for document in df['item7']:
    if isinstance(document, str):
        words = nltk.word_tokenize(document.lower())
        count = 0
        occur_index = []
        occur_list = []
        for word in words:
            if len(word.strip()) != 0:
                count += 1
                for anti_word in anti_list:
                    if word.count(anti_word) > 0:
                        occur_list.append(word)
                        occur_index.append(count)
        position.append(occur_index)
    else:
        position.append([])

avg_position = []
for position_set in position:
    position_list = [item / position_set[-1] for item in position_set[:-1]]
    avg_position.append(sum(position_list) / len(position_list))

df2 = pd.DataFrame(df['item7']).copy()
df2['avg_word_position'] = avg_position
df2.to_csv('../document/18_word_position.csv', index=False)
