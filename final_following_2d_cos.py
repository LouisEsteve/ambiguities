from math import sqrt
import os
import numpy as np

output_uri = '__following_2d_cos.csv'
global_path = 'CORPUS_SPECIAUX/selection'
separator = '\t'
labels = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','SCONJ','VERB','X']

#############################################################################

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

#############################################################################

# THIS FUNCTION IS TO BE VERIFIED
def matrixes_average(a):
    result = [[0.0 for j in i] for i in a[0]]
    # print(f'result size: {len(result)} x {len(result[0])}')
    # print(f'i size: {len(i)} x {len(i[0])}')
    for i in a:
        for j in range(len(i)):
            for k in range(len(i[j])):
                result[j][k] += i[j][k]
    count = len(i)
    for j in range(len(i)):
        for k in range(len(i[j])):
            result[j][k] /= count
    return result

def cosine_similarity(a,b) -> float:
    assert len(a) == len(b)
    # print(type(a[0]),type(b[0]))
    # assert (type(a[0]) == float or type(a[0]) == int) and (type(b[0]) == float or type(b[0]) == int)
    assert (type(a[0]) in [float,np.float32,np.float64]) and (type(b[0]) in [float,np.float32,np.float64])
    part_one = 0.0
    part_two_a = 0.0
    part_two_b = 0.0
    for i in range(len(a)):
        part_one += a[i] * b[i]
        part_two_a += a[i] ** 2
        part_two_b += b[i] ** 2
    # print(type(part_one),part_one.shape)
    # print(type(part_two_a),part_two_a.shape)
    # print(type(part_two_b),part_two_b.shape)
    return (part_one / (sqrt(part_two_a)*sqrt(part_two_b)))

def cosine_similarity_2d_matrix(a,b) -> float:
    # print(len(a),len(b))
    assert len(a) == len(b)
    a_1d = []
    b_1d = []
    """
    for i in range(len(a)):
        assert len(a[i]) == (len(b[i]))
        for j in range(len(a[i])):
            a_1d.append(a[i][j])
            b_1d.append(b[i][j])
    """
    
    for i in range(len(a)):
        a_1d[len(a_1d):] = a[i]
        b_1d[len(b_1d):] = b[i]
    
    # a = [j for j in i for i in a]
    # b = [j for j in i for i in b]
    # return cosine_similarity(a,b)
    return cosine_similarity(a_1d,b_1d)

#############################################################################

_2d_labels = []
for i in labels:
    for j in labels:
        # _2d_labels.append(f'{i} after {j}')
        _2d_labels.append(f'{i} before {j}')

global_table = [
    # ['URI',*_2d_labels]
    ['URI','POS']
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
            if i.startswith('_') and (i.endswith('.conl') or i.endswith('.conll') or i.endswith('.conllu')):
                f = open(lp,'rt',encoding='utf-8')
                d = f.read()
                f.close()
                table = d.split('\n')
                table = [j for j in table if not(j.startswith('#')) and len(j) > 0]
                table = [j.split(separator) for j in table]
                table_pos = [j[3] for j in table]
                

                # global_table.append([lp,*table_pos])
                for j in table_pos:
                    global_table.append([lp,j])
        else:
            recursive(f'{lp}')


# ACTUAL PROCESSING
all_doctypes_table_mean = []
for i in refs:
    # print(i,end='\r')
    ld = os.listdir(i)
    this_doctype_table = []
    # for j in ld:
    for z in ld:
        # if not (j.endswith('.conl') or j.endswith('.conll') or j.endswith('.conllu')):
        if not (z.endswith('.conl') or z.endswith('.conll') or z.endswith('.conllu')):
            continue
        # lp = f'{i}/{j}'
        lp = f'{i}/{z}'
        print(lp,end='\r')
        f = open(lp,'rt',encoding='utf-8')
        d = f.read()
        f.close()
        table = d.split('\n')
        table = [k for k in table if not(k.startswith('#')) and len(k) > 0]
        # table = [k.split(separator) for k in table]
        table = [k.split('\t') for k in table]
        table_pos = [k[3] for k in table]

        __table_pos_updates__ = 0

        __list = np.zeros(shape=(len(labels),len(labels)))
        # print(__list.shape)
        __total_counts = {}
        for j in labels:
            __total_counts[j] = 0
        
        # for j in range(1,len(table_pos)):
        for j in range(len(table_pos)-1):
            try:
                __total_counts[table_pos[j]] += 1
            except:
                pass
                # print(f'[ERROR] Couldn\'t increment __total_counts[{table_pos[j]}]')

            index_current = None
            index_next = None
            try:
                index_current = labels.index(table_pos[j])
                index_next = labels.index(table_pos[j+1])
            except:
                pass
                # print(f'[ERROR] Couldn\'t find labels.index({table_pos[j]}) or labels.index({table_pos[j+1]})')
            
            try:
                if index_current != None and index_next != None:
                    __list[index_current][index_next] += 1
                    __table_pos_updates__ += 1
            except:
                pass

        # CHANGE 2
        assert len(__list) == len(labels)
        for j in range(len(__list)):
            assert len(__list[j]) == len(labels)
            for k in range(len(__list[j])):
                # __list[j][k] /= __total_counts[j]
                # print(__list[j][k],__total_counts[labels[j]],__total_counts[labels[k]])
                
                if __total_counts[labels[j]] != 0:
                    __list[j][k] /= __total_counts[labels[j]]
                
                # if __total_counts[labels[k]] != 0:
                #     __list[j][k] /= __total_counts[labels[k]]
                # print(__list[j][k],labels[j],labels[k])
                assert __list[j][k] <= 1.0
        
        # print(__total_counts)
        this_doctype_table.append(__list)
    # this_doctype_table_mean = np.mean(this_doctype_table,axis=1)

    # this_doctype_table_mean = np.mean(this_doctype_table,axis=2)
    # assert this_doctype_table_mean.shape == (len(labels),len(labels))

    # THIS FIXED IT
    this_doctype_table_mean = matrixes_average(this_doctype_table)
    assert len(this_doctype_table_mean) == len(labels)
    for j in this_doctype_table_mean:
        assert len(j) == len(labels)
    all_doctypes_table_mean.append([i,this_doctype_table_mean])

# for i in all_doctypes_table_mean:
#     print(i[0],i[1])

n = len(all_doctypes_table_mean) ** 2
print(f'{n} pairs')
local_count = 0

final_table = []
# for i in all_doctypes_table_mean:
#     print(i[0],len(i[1]),len(i[1][0]))

for i in all_doctypes_table_mean:
    final_table.append([])
    for j in all_doctypes_table_mean:
        print(f'[{"="*int(local_count/n*20)}{" "*int(20-(local_count/n*20))}] Sim. {i[0]} & {j[0]}',end='\r')
        # print(i[1])
        cos_result = cosine_similarity_2d_matrix(i[1],j[1]) # NO NEED TO MAKE LIST CONVERSION AS WE'RE NOT USING np.array NOW
        # cos_result = cosine_similarity_2d_matrix(i[1].tolist(),j[1].tolist())
        final_table[-1].append(cos_result)
        local_count += 1

##################################################################

# REUSED (WITH MODIFICATIONS) FROM jaccard_lemmas.py

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
    "TUT--CC",
    "TUT--Europarl",
    "TUT--FB",
    "TUT--UDHR"
]
# for i in refs:


# print(t)

x_count = 0
y_count = 0

output_str = ''
# for i in dico.keys():
for i in dico:
    # output_str += '\\node[rotate=90] at ('+ str(x_count+0.5) +',2.0){'+ i +'};\n'
    output_str += '\\node[rotate=90] at ('+ str(x_count+0.5) +',5.0){'+ i +'};\n'
    # output_str += '\\node at (-2.0,'+ str(-x_count-0.5) +'){'+ i +'};\n' # CHANGED y_count TO x_count
    output_str += '\\node at (-5.0,'+ str(-x_count-0.5) +'){'+ i +'};\n' # CHANGED y_count TO x_count
    """
    # for j in dico[i].keys():
    for j in i:
        if j == 'total_count':
            continue
        # output_str += '\\fill[blue!'+ str(final_table[x_count][y_count]*100) +'] ('+ str(x_count) +','+ str(-y_count) +') rectangle ('+ str(x_count+1) +','+ str(-y_count-1) +');\n'
        y_count += 1
    """
    x_count += 1

for i in range(len(final_table)):
    for j in range(len(final_table[i])):
        # output_str += '\\fill[blue!'+ str(min(final_table[i][j]*100,100)) +'] ('+ str(i) +','+ str(-j) +') rectangle ('+ str(i+1) +','+ str(-j-1) +');\n'
        output_str += '\\fill[blue!'+ str(min(final_table[i][j]*100,100)) +'] ('+ str(j) +','+ str(-i) +') rectangle ('+ str(j+1) +','+ str(-i-1) +');\n'

file_name_counter = 0
file_name = 'heatmap_following_' + str(file_name_counter) + '.tex'
while os.path.exists(file_name):
    file_name_counter += 1
    file_name = 'heatmap_following_' + str(file_name_counter) + '.tex'

file = open(file_name,'wt',encoding='utf-8')
file.write(output_str)
file.close()