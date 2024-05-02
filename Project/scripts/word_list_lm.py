import pandas as pd
import re


def sentiment_score(text, sen_list):
    if pd.isnull(text) or text == "":
        return 0
    text = re.sub("[^A-Za-z]+", "", text)
    total_count = sum(text.lower().count(word) for word in sen_list if word in text.lower())
    return total_count / max(len(text), 1)


dataset = pd.read_csv('../document/old/AnnualReports16_processed.csv', encoding='latin-1')
lexicon_LM = pd.read_csv('../word_list/Loughran-McDonald_MasterDictionary_1993-2023.csv')
dataset_cleaned = dataset.dropna()
lexicon_LM['Word'] = lexicon_LM['Word'].str.lower()

Negative = set(lexicon_LM[lexicon_LM['Negative'] != 0]['Word'])
Positive = set(lexicon_LM[lexicon_LM['Positive'] != 0]['Word'])
Uncertainty = set(lexicon_LM[lexicon_LM['Uncertainty'] != 0]['Word'])
Litigious = set(lexicon_LM[lexicon_LM['Litigious'] != 0]['Word'])
Strong_Modal = set(lexicon_LM[lexicon_LM['Strong_Modal'] != 0]['Word'])
Weak_Modal = set(lexicon_LM[lexicon_LM['Weak_Modal'] != 0]['Word'])
Constraining = set(lexicon_LM[lexicon_LM['Constraining'] != 0]['Word'])
Complexity = set(lexicon_LM[lexicon_LM['Complexity'] != 0]['Word'])

for sentiment, score_name in zip(
        [Negative, Positive, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining, Complexity],
        ['Negative_score', 'Positive_score', 'Uncertainty_score', 'Litigious_score', 'Strong_Modal_score',
         'Weak_Modal_score', 'Constraining_score', 'Complexity_score']
):
    dataset_cleaned.loc[:, score_name] = dataset_cleaned['processed_text'].apply(
        lambda x: sentiment_score(x, list(sentiment)))

dataset_cleaned.to_csv('../document/AnnualReports16_lm.csv', index=False)
