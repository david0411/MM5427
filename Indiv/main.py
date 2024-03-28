import nltk
import pandas as pd

nltk.download('punkt')

dataset = pd.read_csv('MM5427_COVID-19_Tweets_2.csv', encoding='latin-1', header=0)
dataset.text = [row.encode('latin-1').decode('utf-8','ignore') for row in dataset.text]
account_tag = []
for text in dataset['text']:
    text_content = text.split(' ')
    account_tag.append(list(filter(lambda x: x.startswith('@'),text_content)))
dataset['account_tag'] = account_tag
print(dataset.head)
