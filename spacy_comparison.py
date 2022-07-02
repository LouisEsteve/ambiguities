import os
import re
import spacy

ambiguities_file = open('AWL_POS_ambiguous_words.csv','rt',encoding='utf-8')
ambiguities_reference_list = ambiguities_file.read().lower().split('\n')
ambiguities_file.close()
del(ambiguities_file)

refs = [
    "gum-master/__dep/GUM_academic",
    "gum-master/__dep/GUM_bio",
    "gum-master/__dep/GUM_conversation",
    "gum-master/__dep/GUM_fiction",
    "gum-master/__dep/GUM_interview",
    "gum-master/__dep/GUM_news",
    "gum-master/__dep/GUM_speech",
    "gum-master/__dep/GUM_textbook",
    "gum-master/__dep/GUM_vlog",
    "gum-master/__dep/GUM_voyage",
    "gum-master/__dep/GUM_whow",
    "CORPUS_SPECIAUX/selection/UD_English-Atis-master",
    "CORPUS_SPECIAUX/selection/UD_English-EWT-master/answers",
    "CORPUS_SPECIAUX/selection/UD_English-EWT-master/email",
    "CORPUS_SPECIAUX/selection/UD_English-EWT-master/newsgroup",
    "CORPUS_SPECIAUX/selection/UD_English-EWT-master/reviews",
    "CORPUS_SPECIAUX/selection/UD_English-EWT-master/weblog",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc2",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc3",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc4",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc6",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc7",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/__doc8",
    "CORPUS_SPECIAUX/selection/UD_English-LinES-master/_m_doc1",
    "CORPUS_SPECIAUX/selection/TUT-CoNLL/CC",
    "CORPUS_SPECIAUX/selection/TUT-CoNLL/Europarl",
    "CORPUS_SPECIAUX/selection/TUT-CoNLL/FB",
    "CORPUS_SPECIAUX/selection/TUT-CoNLL/UDHR"
]

dico = [
    "GUM--academic",
    "GUM--bio",
    "GUM--conversation",
    "GUM--fiction",
    "GUM--interview",
    "GUM--news",
    "GUM--speech",
    "GUM--textbook",
    "GUM--vlog",
    "GUM--voyage",
    "GUM--whow",
    "ATIS",
    "EWT--answer",
    "EWT--email",
    "EWT--newsgroup",
    "EWT--review",
    "EWT--weblog",
    "LinES--doc2",
    "LinES--doc3",
    "LinES--doc4",
    "LinES--doc6",
    "LinES--doc7",
    "LinES--doc8",
    "LinES-m-doc1",
    "ParTUT--CC",
    "ParTUT--Europarl",
    "ParTUT--FB",
    "ParTUT--UDHR"
]

dico_count = 0

# space_after_is_no_pattern = 'SpaceAfter=[nN]o'
# space_after_is_no_regex = re.compile(space_after_is_no_pattern)

# text_pattern = '# text = ([^\\n]+)'
# text_regex = re.compile(text_pattern)

csv_table = [
    ['doctype','doctype_latex','nos_pos_amb','pos_amb']
]

# nlp = spacy.Pipeline(lang='en',processors='tokenize,mwt,pos')
# nlp = spacy.Pipeline(lang='en',processors='tokenize,mwt,pos',tokenize_pretokenized=True)
nlp = spacy.load('en_core_web_trf')

for i in refs:
    ref_upper_count = 0
    ref_lower_count = 0
    ref_pos_amb_upper_count = 0
    ref_pos_amb_lower_count = 0
    ref_not_pos_amb_upper_count = 0
    ref_not_pos_amb_lower_count = 0
    token_pos_pairs = []
    ld = os.listdir(i)
    for j in ld:
        if not (j.endswith('.conl') or j.endswith('.conll') or j.endswith('.conllu')):
            continue
        lp = f'{i}/{j}'
        f = open(lp,'rt',encoding='utf-8')
        d = f.read()
        f.close()
        table = d.split('\n')
        table = [k.split('\t') for k in table if len(k) > 0 and not k.startswith('#')]
        # text = ''
        # for k in table:
        #     text = f'{text} {k[1]}'
        # print(text)
        words = [k[1] for k in table]
        doc = spacy.tokens.Doc(nlp.vocab,words=words)
        # spacy_pred = nlp(text)
        # spacy_pred = spacy_pred.to_dict()
        spacy_pred = nlp(doc)
        # spacy_pred = [[q for q in k] for k in spacy_pred]
        """
        __spacy_pred = []
        for k in spacy_pred:
            for q in k:
                __spacy_pred.append(q)
        spacy_pred = __spacy_pred
        """
        # print(spacy_pred[:5])
        # print(len(table),len(spacy_pred))
        upper_count = 0
        lower_count = 0
        pos_amb_upper_count = 0
        pos_amb_lower_count = 0
        not_pos_amb_upper_count = 0
        not_pos_amb_lower_count = 0
        index_diff = 0
        
        for k in spacy_pred:
            try:
                # while k['text'] != table[lower_count+index_diff][1]:
                while k.text != table[lower_count+index_diff][1]:
                # while not (k['text'].startswith(table[lower_count+index_diff][1]) or table[lower_count+index_diff][1].startswith(k['text'])):
                    # print(k['text'],table[lower_count+index_diff][1])
                    print(k.text,table[lower_count+index_diff][1])
                    index_diff += 1
                # if k['upos'] == table[lower_count+index_diff][3]:
                if k.pos_ == table[lower_count+index_diff][3]:
                    # if k['text'] in ambiguities_reference_list:
                    if k.text.lower() in ambiguities_reference_list:
                        pos_amb_upper_count += 1
                    else:
                        not_pos_amb_upper_count += 1
                    upper_count += 1
                lower_count += 1
                # if k['text'] in ambiguities_reference_list:
                if k.text.lower() in ambiguities_reference_list:
                    pos_amb_lower_count += 1
                else:
                    not_pos_amb_lower_count += 1
            except:
                print('BROKE')
                break
        """
        for k in table:
            try:
                # while k['text'] != table[lower_count+index_diff][1]:
                while not (spacy_pred[lower_count+index_diff]['text'].startswith(k[1]) or k[1].startswith(spacy_pred[lower_count+index_diff]['text'])):
                    print(k['text'],table[lower_count+index_diff][1])
                    index_diff += 1
                if k['upos'] == table[lower_count+index_diff][3]:
                    upper_count += 1
                lower_count += 1
            except:
                print('BROKE')
                break
        """
        # print(index_diff)
        # print(f'{j} : {upper_count}/{lower_count} ; {(upper_count/lower_count)*100}%')
        try:
            if pos_amb_lower_count == 0:
                print(f'{j}: not_pos_amb -> {(not_pos_amb_upper_count/not_pos_amb_lower_count)*100}%\tpos_amb -> NO POS AMBIGUITIES',end='\r')
            else:
                print(f'{j}: not_pos_amb -> {(not_pos_amb_upper_count/not_pos_amb_lower_count)*100}%\tpos_amb -> {(pos_amb_upper_count/pos_amb_lower_count)*100}%',end='\r')
        except:
            pass
        ref_upper_count += upper_count
        ref_lower_count += lower_count
        ref_pos_amb_upper_count += pos_amb_upper_count
        ref_pos_amb_lower_count += pos_amb_lower_count
        ref_not_pos_amb_upper_count += not_pos_amb_upper_count
        ref_not_pos_amb_lower_count += not_pos_amb_lower_count
    try:
        print(f'ref. {i}: not_pos_amb -> {(ref_not_pos_amb_upper_count/ref_not_pos_amb_lower_count)*100}%\tpos_amb -> {(ref_pos_amb_upper_count/ref_pos_amb_lower_count)*100}%')
    except:
        pass
    csv_table.append([i,dico[dico_count],str(ref_not_pos_amb_upper_count/ref_not_pos_amb_lower_count),str(ref_pos_amb_upper_count/ref_pos_amb_lower_count)])
    dico_count += 1

csv_table = ['\t'.join(i) for i in csv_table]
csv_table = '\n'.join(csv_table)
f = open('spacy_results.csv','wt',encoding='utf-8')
f.write(csv_table)
f.close()
print('Wrote spacy_results.csv')
