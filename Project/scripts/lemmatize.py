import re
import pandas as pd
import spacy
import gensim
import time


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=None):
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


dataset = pd.read_csv('../document/AnnualReports16_processed.csv', header=0)
item7 = dataset['processed_text'].values.tolist()
print("Preprocessing")
tic = time.perf_counter()
for i in range(len(item7)):
    if isinstance(item7[i], str):
        item7[i] = re.sub("[^A-Za-z0-9]+", " ", item7[i])
        item7[i] = item7[i].lower()
toc = time.perf_counter()
print(f"Done preprocessing in {toc - tic:0.1f} seconds")

item7_words = list(sent_to_words(item7))

bigram = gensim.models.Phrases(item7_words, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[item7_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print("Making Bigrams")
tic = time.perf_counter()
item7_words_bigrams = make_bigrams(item7_words)
toc = time.perf_counter()
print(f"Done bigrams in {toc - tic:0.1f} seconds")

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

print("Lemmatizing text")
tic = time.perf_counter()
item7_words_lemmatized = lemmatization(item7_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
toc = time.perf_counter()
print(f"Done lemmatization in {toc - tic:0.1f} seconds")
dataset['lemmatized_text'] = item7_words_lemmatized
dataset.to_csv('../document/AnnualReports16_lemmatized.csv', index=False)
