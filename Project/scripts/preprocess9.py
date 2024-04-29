import re
import pandas as pd
from nltk import word_tokenize

delimiter = ' '

dataset = pd.read_csv('../document/AnnualReports1618.csv', header=0)
dataset_2016 = dataset[dataset['filed_date'] < 20170000]
processed_text = []

for line in dataset_2016['item7']:
# for i in range(10):
    if isinstance(line, str):
        content = str(re.sub("[^A-Za-z]+", " ", line)).lower()
        tokens = word_tokenize(content)
        processed_text.append(delimiter.join(tokens))
    else:
        processed_text.append('')
dataset_2016['processed_text'] = processed_text
dataset_2016.drop('item7', axis=1, inplace=True)
dataset_2016.to_csv('../document/AnnualReports16_processed9.csv', index=False)
