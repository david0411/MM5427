import pandas as pd
import nltk
from spacy.lang.en import English
from nltk.stem import WordNetLemmatizer


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


df = pd.read_csv('../document/AnnualReports16_processed2.csv')
df['sentence_count'] = df['item7'].apply(count_sentences)
df = df[df['sentence_count'] > 10]
df['item7'] = df['item7'].apply(delete_first_sentence)
df['item7'] = df['item7'].replace('\n', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = df['item7'].replace('[\d.,]+|[^\w\s]', '', regex=True)
df['item7'] = [x.lower() for x in df['item7']]
df['item7'] = df['item7'].replace('item 7.', '', regex=True)

documents = []
stemmer = WordNetLemmatizer()

for line in df['item7']:
    nlp = English()
    my_doc = nlp(line)
    token_list = []

    for token in my_doc:
        token_list.append(token.text)
    filtered_sentence = []

    for word in token_list:
        lexeme = nlp.vocab[word]
        if not lexeme.is_stop:
            filtered_sentence.append(word)

    document = [stemmer.lemmatize(word) for word in filtered_sentence]
    document = ' '.join(document)
    documents.append(document)

df['item7'] = documents

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

sen_df['pos_anti_increment'] = (sen_df['Pos_Dic'] + sen_df['Anti_Dic']) / sen_df['Anti_Dic']
sen_df['neg_anti_increment'] = (sen_df['Anti_Dic'] - sen_df['Neg_Dic']) / sen_df['Anti_Dic']

sen_df['result'] = sen_df.apply(
    lambda row: row['pos_anti_increment'] if row['Pos_Dic'] > row['Neg_Dic'] else row['neg_anti_increment'], axis=1)
sen_df.to_csv('../document/emot_score_16.csv', index=False)
