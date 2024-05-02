import nltk
import pandas as pd

temp = 0
word_list = ['retire', 'resign', 'leave', 'transition', 'reinvent', 'reskill', 'pivot', 'upskill', 'cross-train',
             'network', 'retire', 'resign', 'transition out', 'career break', 'switch gears', 'change tracks',
             'redeploy', 'relocate', 'restart', 'redirection', 'redirection', 'transformation', 'new beginning',
             'second act', 'encore career', 'portfolio career', 'side hustle', 'gig work', 'freelance', 'consult',
             'downshift', 'recharge', 'sabbatical', 'self-employment', 'entrepreneurship']
dataset = pd.read_csv('../../document/old/AnnualReports16_processed.csv', header=0)
dataset = dataset.dropna(axis=0)
for line in dataset['processed_text']:
    words = nltk.word_tokenize(line)
    for w in word_list:
        temp += words.count(w)
print(temp)
# word_ = wordnet.synsets("leave")
# print(word_)
