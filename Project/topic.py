import re
import pandas as pd
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # don't skip this
import matplotlib.pyplot as plt


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


dataset = pd.read_csv('AnnualReports16_processed.csv', header=0)
text = dataset['processed_text'].values.tolist()
for i in range(len(text)):
    if isinstance(text[i], str):
        text[i] = re.sub("[^A-Za-z0-9]+", " ", text[i])
        text[i] = text[i].lower()

text_words = list(sent_to_words(text))

bigram = gensim.models.Phrases(text_words, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[text_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

review_words_bigrams = make_bigrams(text_words)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
text_lemmatized = lemmatization(review_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(text_lemmatized)
corpus = [id2word.doc2bow(text) for text in text_lemmatized]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=9, update_every=1,
                                            chunksize=100, passes=10, alpha='auto', per_word_topics=True)
print('Perplexity: ', lda_model.log_perplexity(corpus))
coherence_model_lda = CoherenceModel(model=lda_model, texts=text_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


pyLDAvis.enable_notebook()
print(gensimvis.prepare(lda_model, corpus, id2word))

topic = []

for i, row_list in enumerate(lda_model[corpus]):
    row = sorted(row_list[0], key=lambda x: (x[1]), reverse=True)
    topic.append(row[0][0])

dataset['topic'] = topic
