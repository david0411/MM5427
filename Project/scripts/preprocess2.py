import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize

delimiter = ' '
stop_words = set(stopwords.words('english'))

x = pd.read_csv('../document/AnnualReports1618.csv', header=0)
x_2016 = x[x['filed_date'] < 20170000]

x_2016.to_csv('../document/AnnualReports16_processed2.csv', index=False)
