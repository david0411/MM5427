import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize

delimiter = ' '
stop_words = set(stopwords.words('english'))

dataset = pd.read_csv('../document/AnnualReports1618.csv', header=0)
dataset_2016 = dataset[dataset['filed_date'] < 20170000]
processed_text = []

for i in range(dataset_2016.shape[0]):
    # for i in range(10):
    content = str(dataset_2016['item7'][i]).lower()
    if content != 'nan':
        discard_set = {'$', '%', '(', ')', ''}
        tokens = word_tokenize(content)
        tokens = [w for w in tokens if not w in stop_words]
        for item in tokens:
            for word in re.findall(r"\d+(?:\.\d+)?", item):
                discard_set.add(word)
        tokens = [w for w in tokens if not w in discard_set]
        processed_text.append(delimiter.join(tokens))
    else:
        processed_text.append('')
dataset_2016['processed_text'] = processed_text
dataset_2016.drop('item7', axis=1, inplace=True)
dataset_2016.to_csv('../document/AnnualReports16_processed.csv', index=False)