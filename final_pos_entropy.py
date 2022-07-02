import os

from entropy import entropy

output_uri = '__entropy_recursive_GUM.csv'
global_path = 'gum-master/__dep'
# output_uri = '__entropy_recursive.csv'
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
    ['URI','H(X)']
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
            if (i.startswith('_') or 'TUT-CoNLL' in lp) and (i.endswith('.conl') or i.endswith('.conll') or i.endswith('.conllu')):
                f = open(lp,'rt',encoding='utf-8')
                d = f.read()
                f.close()
                table = d.split('\n')
                table = [j for j in table if not(j.startswith('#')) and len(j) > 0]
                table = [j.split(separator) for j in table]
                table_pos = [j[3] for j in table]
                di = {}
                """
                for j in range(1,len(table_pos)):
                    if table_pos[j] not in di.keys():
                        di[table_pos[j]] = {
                            '__total_count__': 0
                        }
                    di[table_pos[j]]['__total_count__'] += 1

                    if table_pos[j-1] not in di[table_pos[j]].keys():
                        di[table_pos[j]][table_pos[j-1]] = 0
                    di[table_pos[j]][table_pos[j-1]] += 1
                
                # lt = [lp,*[0 for j in _2d_labels]]
                lt = [lp,*_2d_labels]
                for j in di.keys():
                    for k in di[j].keys():
                        if k == '__total__count':
                            continue
                        try:
                            index = -1
                            # index = _2d_labels.index(f'{j} after {k}')
                            index = lt.index(f'{j} after {k}')
                            lt[index] = di[j][k] / di[j]['__total_count__']
                        except:
                            pass
                for j in range(1,len(lt)):
                    # if ' after ' in lt[j]:
                    if type(lt[j]) == str:
                        lt[j] = 0
                global_table.append(lt)
                """




                """
                # global_table.append([lp,*table_pos])
                for j in table_pos:
                    global_table.append([lp,j])
                """

                global_table.append([lp,str(entropy(table_pos))])
        else:
            recursive(f'{lp}')


recursive(global_path)
global_table = [[str(i) for i in j] for j in global_table]
global_table = [';'.join(i) for i in global_table]
global_table = '\n'.join(global_table)

f = open(output_uri,'wt',encoding='utf-8')
f.write(global_table)
f.close()
