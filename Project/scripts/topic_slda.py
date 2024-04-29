import pandas as pd
import ast
import tomotopy as tp

dataset = pd.read_csv('../document/AnnualReports16_lemmatized.csv', header=0)
dataset = dataset[dataset['lemmatized_text'] != "[]"]

slda_model = tp.SLDAModel(k=5, eta=0.01, min_cf=5, rm_top=0)  # Top Word: 1. increase 2. expense
for line in dataset['lemmatized_text'].apply(ast.literal_eval):
    line = [item for item in line if item not in ['include', 'company']]
    slda_model.add_doc(line)
slda_model.train(iter=1000, show_progress=True)

# Get the most probable words for each topic
top_words_per_topic = []
for topic_id in range(slda_model.k):
    words = slda_model.get_topic_words(topic_id, top_n=10)
    top_words_per_topic.append(words)
    print("Topic {}: {}".format(topic_id + 1, words))

slda_model.save('../model/slda_model.bin')
