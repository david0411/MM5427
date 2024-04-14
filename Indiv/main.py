import re
import unicodedata
import string
import pandas as pd
import nltk
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')

#%% md
# Load data file with header
#%%
dataset = pd.read_csv('MM5427_COVID-19_Tweets_2.csv', encoding='latin-1', header=0)
dataset.text = [row.encode('latin-1').decode('utf-8', 'ignore') for row in dataset.text]
#%% md
# 1.I
#%% md
## Construct a list acc_tag that contain all account tags for each tweet
#%%
acc_tag = []
for text in dataset['text']:
    # split the text by whitespace
    text_content = re.split(r'\s|[(),.;!?/\'"]', text)
    # filter the list of items that starts with "@" and append to the acc_tag list
    acc_tag.append(list(filter(lambda x: x.startswith('@') and len(x) > 1, text_content)))
# add the acc_tag as the new column
dataset['acc_tag'] = acc_tag
#%% md
## Do the same for hashtag and URL
#%%
hashtag = []
for text in dataset['text']:
    text_content = re.split(r'\s|[(),.;!?/\'"]', text)
    hashtag.append(list(filter(lambda x: x.startswith('#') and len(x) > 1, text_content)))
dataset['hashtag'] = hashtag
#%%
URL = []
for text in dataset['text']:
    text_content = re.split(r'\s', text)
    URL.append(list(filter(lambda x: x.startswith('http'), text_content)))
dataset['URL'] = URL
#%%
dataset.head()
#%% md
# 1.II
#%%
# put the acc_tag sublist into one single list
acc_tag_single_list = [item for sublist in acc_tag for item in sublist]
# count the number of each acc in the list
acc_counter = Counter(acc_tag_single_list)
# get the top 10 countered acc
top_10_acc = acc_counter.most_common(10)
print(top_10_acc)
#%% md
# The result shows that country leaders and international health organizations are the most tagged.
# This may reflect people seeking information, express opinion or emotion towards authority
#%% md
## Do the same for hashtag and URL
#%%
hashtag_single_list = [item for sublist in hashtag for item in sublist]
hash_counter = Counter(hashtag_single_list)
top_10_hash = hash_counter.most_common(10)
print(top_10_hash)
#%% md
## All 10 hashtags are related to COVID
#%%
URL_single_list = [item for sublist in URL for item in sublist]
URL_counter = Counter(URL_single_list)
top_10_URL = URL_counter.most_common(10)
print(top_10_URL)
#%% md
## Those URL are mainly news related. Showing people wants to spread the information in the social media
#%% md
## Define a function for removing punctuation
#%%
def remove_punctuation(input_string):
    # Create a translation table mapping punctuation characters to empty string
    translator = str.maketrans('', '', string.punctuation + '‘’“”–•・❝❞')

    # Transform the full-width characters to half-with characters
    normalized_text = unicodedata.normalize('NFKC', input_string)
    # Remove punctuation using the translation table
    no_punct = normalized_text.translate(translator)

    return no_punct
#%% md
## Use the result in 1.I to do the removal
#%%
processed_text = []
delimiter = ' '
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
for i in range(dataset.shape[0]):
    # get the data of the row
    text = dataset['text'][i]
    acc_tag = dataset['acc_tag'][i]
    hashtag = dataset['hashtag'][i]
    URL = dataset['URL'][i]
    # remove account tags
    for tag1 in acc_tag:
        text = text.replace(tag1, '')
    # remove hashtag
    for tag2 in hashtag:
        text = text.replace(tag2, '')
    # remove URL
    for tag3 in URL:
        text = text.replace(tag3, '')
    # remove punctuations by above function
    text = remove_punctuation(text)
    # tokenize the text, remove the stop words and join the tokens to sentence again
    tokens = word_tokenize(text)
    processed_text.append(delimiter.join([w for w in tokens if not w in stop_words]))
# replace the original text with processed one
dataset['text'] = processed_text
#%%
dataset.head()
#%% md
# 1.IV
#%%
lower_text = []
for i in range(dataset.shape[0]):
    # apply lowercase for each row
    lower_text.append(dataset['text'][i].lower())
# replace the text
dataset['text'] = lower_text
#%% md
# 1.V
#%%
stem_text = []
delimiter = ' '
stemmer = PorterStemmer()
for i in range(dataset.shape[0]):
    stem_token = []
    # tokenize the text and apply stemming for each token
    for tokens in word_tokenize(dataset['text'][i]):
        stem_token.append(stemmer.stem(tokens))
    stem_text.append(delimiter.join(stem_token))
    print(stem_text)
    exit()
# replace the text
dataset['text'] = stem_text
#%%
dataset.head()
#%% md
# 1.VI
#%% md
## Finding the emoji. I notice there are some emoji stuck together.
## I tried both considering it to be one emoji and multiple emoji
#%%
emoticons1 = []
emoticons2 = []
for i in range(dataset.shape[0]):
    text = dataset['text'][i]
    # Consider consecutive emoji as one emoji
    emoticons1.append(re.findall('[\U0001F600-\U0001F64F]+', text))
    # Consider consecutive emoji as multiple emoji
    emoticons_multi = re.finditer('[\U0001F600-\U0001F64F]', text)
    for emoji in emoticons_multi:
        emoticons2.append(emoji.group())
# create single list of emoji for counting
emoticons_single_list1 = [item for sublist in emoticons1 for item in sublist]
emoticons_counter1 = Counter(emoticons_single_list1)
top_3_emoticons1 = emoticons_counter1.most_common(3)
print(top_3_emoticons1)

emoticons_counter2 = Counter(emoticons2)
top_3_emoticons2 = emoticons_counter2.most_common(3)
print(top_3_emoticons2)