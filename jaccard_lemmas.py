import os
from jaccard import jaccard

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

q = {}

for i in refs:
    # print(i,end=' '*len(i)+'\r')
    print(i,end='\r')
    s = []
    ld = os.listdir(i)
    for j in ld:
        p = i+'/'+j
        if os.path.isdir(p) or not (j.endswith('.conl') or j.endswith('.conll') or j.endswith('.conllu')):
            continue
        # print(p,end=' '*len(p)+'\r')
        print(p,end='\r')
        f = open(p,'rt',encoding='utf-8')
        d = f.read()
        f.close()
        d = d.split('\n')
        d = [k for k in d if len(k) > 0 and k[0]!='#']
        d = [k.split('\t') for k in d]
        # d = [k[2] for k in d]
        d = [k[2].lower() for k in d]
        # s[len(s):] = d
        for k in d:
            if k not in s:
                s.append(k)
    q[p] = s

t = []
for i in q.keys():
    t.append([])
    for j in q.keys():
        t[-1].append(jaccard(q[i],q[j]))

final_table = t
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
file_name = 'heatmap_jaccard_lemmas_' + str(file_name_counter) + '.tex'
while os.path.exists(file_name):
    file_name_counter += 1
    file_name = 'heatmap_jaccard_lemmas_' + str(file_name_counter) + '.tex'

file = open(file_name,'wt',encoding='utf-8')
file.write(output_str)
file.close()