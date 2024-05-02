import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from spacy.lang.en import English


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
df['item7'] = df['item7'].apply(delete_first_sentence)
df['item7'] = df['item7'].replace('\n', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = df['item7'].replace('[\d.,]+|[^\w\s]', '', regex=True)
df['item7'] = [x.lower() for x in df['item7']]
df['item7'] = df['item7'].replace('item 7.', '', regex=True)

documents = []

stemmer = WordNetLemmatizer()

for text in df['item7']:
    nlp = English()
    my_doc = nlp(text)
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    filtered_sentence = []

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)

    document = [stemmer.lemmatize(word) for word in filtered_sentence]
    document = ' '.join(document)

    documents.append(document)

df['item7'] = documents

df_16 = df[df['filed_date'] < 20170000]
df_17 = df[(df['filed_date'] > 20170000) & df['filed_date'] < 20180000]
df_18 = df[df['filed_date'] > 20180000]

df_16.to_csv('../document/Preprocessed_16.csv', index=False)
df_17.to_csv('../document/Preprocessed_17.csv', index=False)
df_18.to_csv('../document/Preprocessed_18.csv', index=False)
