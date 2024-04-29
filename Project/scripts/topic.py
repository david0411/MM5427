import pandas as pd
import gensim
import time
import ast
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # don't skip this
import matplotlib.pyplot as plt


dataset = pd.read_csv('../document/AnnualReports16_lemmatized.csv', header=0)
dataset = dataset[dataset['lemmatized_text'] != "[]"]
item7_words_lemmatized = dataset['lemmatized_text'].apply(ast.literal_eval).to_list()

id2word = corpora.Dictionary(item7_words_lemmatized)
corpus = [id2word.doc2bow(text) for text in item7_words_lemmatized]

print("Building Model")
tic = time.perf_counter()
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=9, update_every=1,
                                            chunksize=100, passes=10, alpha='auto', per_word_topics=True)
toc = time.perf_counter()
print(f"Done lda model in {toc - tic:0.1f} seconds")

print('Perplexity: ', lda_model.log_perplexity(corpus))

# coherence_model_lda = CoherenceModel(model=lda_model, texts=item7_words_lemmatized,
#                                      dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score: ', coherence_lda)

lda_data = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(lda_data, 'LDA.html')

topic = []

for i, row_list in enumerate(lda_model[corpus]):
    row = sorted(row_list[0], key=lambda x: (x[1]), reverse=True)
    topic.append(row[0][0])

dataset['topic'] = topic
dataset.to_csv('../document/AnnualReports16_topic.csv', index=False)
