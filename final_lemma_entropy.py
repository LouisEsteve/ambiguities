import os

from entropy import entropy

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! THIS HAS BEEN REMODELLED FOR LEMMAS !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

output_uri = 'lemma_entropy_recursive_GUM.csv'
global_path = 'gum-master/__dep'
# output_uri = 'lemma_entropy_recursive.csv'
# global_path = 'CORPUS_SPECIAUX/selection'
separator = '\t'
labels = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','SCONJ','VERB','X']

#############################################################################

_2d_labels = []
for i in labels:
    for j in labels:
        _2d_labels.append(f'{i} after {j}')

global_table = [
    # ['URI',*_2d_labels]
    # ['URI','H(X)']
    ['URI','H_lemmas(X)']
]

def recursive(p):
    # if os.path.isfile(p):
    #     if p.startswith('_') and p.en
    listdir = os.listdir(p)
    for i in listdir:
        lp = f'{p}/{i}'
        if lp.startswith('.'):
            continue
        if os.path.isfile(lp):
            # if (i.startswith('_') or 'TUT-CoNLL' in lp) and (i.endswith('.conl') or i.endswith('.conll') or i.endswith('.conllu')):
            if (i.endswith('.conl') or i.endswith('.conll') or i.endswith('.conllu')):
                # MOD
                print(i,end='\r')
                f = open(lp,'rt',encoding='utf-8')
                d = f.read()
                f.close()
                table = d.split('\n')
                table = [j for j in table if not(j.startswith('#')) and len(j) > 0]
                table = [j.split(separator) for j in table]
                # MOD
                # table_pos = [j[3] for j in table] # POS
                table_lemma = [j[2].lower() for j in table] # LEMMAS
                

                global_table.append([lp,str(entropy(table_lemma))])
        else:
            recursive(f'{lp}')


recursive(global_path)
global_table = [[str(i) for i in j] for j in global_table]
global_table = [';'.join(i) for i in global_table]
global_table = '\n'.join(global_table)

f = open(output_uri,'wt',encoding='utf-8')
f.write(global_table)
f.close()
