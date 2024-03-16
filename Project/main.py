import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


word_count_list =[]
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

x = pd.read_csv('AnnualReports1618.csv', header=0)
x_2016 = x[x['filed_date'] < 20170000]

for i in range(x_2016.shape[0]):
# for i in range(20):
    content = str(x_2016['item7'][i]).lower()
    if content != 'nan':
        words = set(re.split('[ (),.]', content))
        discard_set = set()
        words_swr = set(w for w in words if not w in stop_words)
        for item in words_swr:
            for num in re.findall(r"\d+", item):
                discard_set.add(num)
            if '$' in item or '%' in item or item == '':
                discard_set.add(item)
        for item in discard_set:
            words_swr.discard(item)
        word_count_list.append([x_2016['company_name'][i], len(words_swr)])
pd.DataFrame(word_count_list, columns=['Company Name', 'Word Count']).to_excel('word_count.xlsx', index=False)
