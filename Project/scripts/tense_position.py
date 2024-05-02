import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag


def count_sentences(text):
    if pd.isnull(text):
        return 0
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def delete_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) > 2:
        return ' '.join(sentences[2:])
    else:
        return text


def tense_detect(tagged_sentence):
    verb_tags = ['MD', 'MDF',
                 'BE', 'BEG', 'BEN', 'BED', 'BEDZ', 'BEZ', 'BEM', 'BER',
                 'DO', 'DOD', 'DOZ',
                 'HV', 'HVG', 'HVN', 'HVD', 'HVZ',
                 'VB', 'VBG', 'VBN', 'VBD', 'VBZ',
                 'SH',
                 'TO',

                 'JJ'  # maybe?
                 ]

    verb_phrase = []
    for item in tagged_sentence:
        if item[1] in verb_tags:
            verb_phrase.append(item)

    grammar = r'''
            Future_Perfect_Continuous:             {<MD><VB><VBN><VBG>}
            Future_Continuous:                     {<MD><VB><VBG>}
            Future_Perfect:                        {<MD><VB><VBN>}
            Future_Indefinite:                     {<MD><VB>}
            future perfect continuous passive:     {<MDF><HV><BEN><BEG><VBN|VBD>+}
            future perfect continuous:             {<MDF><HV><BEN><VBG|HVG|BEG>+}   
            future perfect passive:                {<MDF><HV><BEN><VBN|VBD>+}   
            future perfect:                        {<MDF><HV><HVN|BEN|VBN|VBD>+}   
            future continuous passive:             {<MDF><BE><BEG><VBN|VBD>+}   
            future continuous:                     {<MDF><BE><VBG|HVG|BEG>+}   
            future indefinite passive:             {<MDF><BE><VBN|VBD>+}
            future indefinite:                     {<MDF><BE|DO|VB|HV>+}       
            '''

    cp = nltk.RegexpParser(grammar)
    tenses_set = set()
    if len(verb_phrase) >= 1:
        result = cp.parse(verb_phrase)
        for node in result:
            if type(node) is nltk.tree.Tree:
                tenses_set.add(node.label())
    return tenses_set


df = pd.read_csv('../document/AnnualReports_18.csv')
df['sentence_count'] = df['item7'].apply(count_sentences)
df = df[df['sentence_count'] > 10]
df['item7'] = df['item7'].apply(delete_first_sentence)
df['item7'] = df['item7'].replace('\n', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = df['item7'].replace('\r', '', regex=True)
df['item7'] = [x.lower() for x in df['item7']]
df['item7'] = df['item7'].replace('item 7.', '', regex=True)

position = []
for document in df['item7']:
    if isinstance(document, str):
        sentences = nltk.sent_tokenize(document.lower())
        count = 0
        occur_index = []
        for sentence in sentences:
            count += 1
            if len(sentence.strip()) != 0:
                text = word_tokenize(sentence)
                tagged = pos_tag(text)
                tense_data = tense_detect(tagged)
                if len(tense_data) > 0:
                    occur_index.append(count)
        occur_index.append(count)
        position.append(occur_index)
    else:
        position.append([])

avg_position = []
for position_set in position:
    position_list = [item / position_set[-1] for item in position_set[:-1]]
    if len(position_list) > 0:
        avg_position.append(sum(position_list) / len(position_list))
    else:
        avg_position.append(0)
df2 = pd.DataFrame(df['item7']).copy()
df2['avg_tense_position'] = avg_position
df2.to_csv('../document/18_tense_position.csv', index=False)
