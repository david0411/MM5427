import pandas as pd
import re
import math


def tense_count(text, tense_list):
    f_list = []
    for t in text:
        f = 0
        for w in tense_list:
            pattern = w.replace('*', '.*')  # 将*替换为.*
            regex = re.compile(pattern)
            f += len(regex.findall(t))
        f_list.append(f)
    return f_list


def count_words_without_punctuation(text):
    # 计算单词总数
    word_count = len(text.split())
    return word_count


df = pd.read_csv('../document/Preprocessed_18.csv')
lexicon = pd.read_csv('../word_list/Future.txt', sep='\t', names=['term', 'category', 'associated'])

lexicon['term'] = lexicon['term'].str.lower()

future_list = list(lexicon[(lexicon['category'] == 'future') & (lexicon['associated'] == 1)].term)
past_list = list(lexicon[(lexicon['category'] == 'past') & (lexicon['associated'] == 1)].term)
present_list = list(lexicon[(lexicon['category'] == 'present') & (lexicon['associated'] == 1)].term)
positive_list = list(lexicon[(lexicon['category'] == 'positive') & (lexicon['associated'] == 1)].term)
negative_list = list(lexicon[(lexicon['category'] == 'negative') & (lexicon['associated'] == 1)].term)


df2 = pd.DataFrame(df['item7']).copy()
df2['future_count'] = tense_count(df2['item7'], future_list)
df2['past_count'] = tense_count(df2['item7'], past_list)
df2['present_count'] = tense_count(df2['item7'], present_list)
df2['positive_count'] = tense_count(df2['item7'], positive_list)
df2['negative_count'] = tense_count(df2['item7'], negative_list)
df2['item7'] = df2['item7'].astype(str)
df2['word_counts'] = df2['item7'].apply(count_words_without_punctuation)

df2['percent_of_future_words'] = [(100 * df2.future_count[i] / df2.word_counts[i]) for i in range(len(df2.word_counts))]
df2['percent_of_past_words'] = [(100 * df2.past_count[i] / df2.word_counts[i]) for i in range(len(df2.word_counts))]
df2['percent_of_present_words'] = [(100 * df2.present_count[i] / df2.word_counts[i]) for i in range(len(df2.word_counts))]
df2['percent_of_positive_words'] = [(100 * df2.positive_count[i] / df2.word_counts[i]) for i in range(len(df2.word_counts))]
df2['percent_of_negative_words'] = [(100 * df2.negative_count[i] / df2.word_counts[i]) for i in range(len(df2.word_counts))]

FvsP = []
PvsN = []

for row in range(len(df2)):
    FvsP_in_row = math.log(
        (1 + df2.percent_of_future_words[row]) / (1 + df2.percent_of_present_words[row] + df2.percent_of_past_words[row]))
    FvsP.append(FvsP_in_row)
    PvsN_in_row = math.log((1 + df2.percent_of_positive_words[row]) / (1 + df2.percent_of_negative_words[row]))
    PvsN.append(PvsN_in_row)

df2['FvsP'] = FvsP
df2['PvsN'] = PvsN

df2.to_csv('../document/18_tense_index.csv', index=False)
