import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag


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


dataset = pd.read_csv('../document/AnnualReports16_processed2.csv')
position = []
for document in dataset['item7']:
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
dataset_2016['position'] = position
dataset_2016.to_csv('../document/AnnualReports16_grammar_position.csv', index=False)
