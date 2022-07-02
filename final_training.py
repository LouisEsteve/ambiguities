from os.path import exists
from os import listdir
# from sys import getsizeof
import re
# import time
from time import time, ctime
import numpy as np
from gensim.models import KeyedVectors
from entropy import entropy

import joblib

from sklearn.neural_network import MLPClassifier
# from sklearn import svm
from sklearn.svm import LinearSVC
# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
# from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.pipeline import Pipeline
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import learning_curve
from sklearn.feature_extraction import FeatureHasher


# display_confusion_matrix = True
# display_pos_ambiguities_confusion_matrix = True
display = False

if display:
    # from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import ConfusionMatrixDisplay
    from matplotlib import pyplot as plt
    # from matplotlib import cm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# from math import sqrt

# global_t = time.time()
global_t = time()

print('='*64)
# print(time.ctime())
print(ctime())
del(ctime)
print(__name__)
print('='*32)


saving = True
training = True
pos_ambiguities_only = False
# pos_ambiguities_count = 0
pos_ambiguities_X_train = []
pos_ambiguities_y_train = []

# <PARSING>

directory_path = 'gum-master/dep/'

# line_regex = re.compile()

# sentence_index = 0
documents = []
current_document = []

# punct_count = 0

entropy_csv = open('entropy_csv.csv','wt',encoding='utf-8')
entropy_csv.write('file_name;reference_entropy\n')
entropy_csv.close()

def parse_conllu(uri):
    if exists(uri):
        # current_punct_count = 0
        """
        with open(uri,'rt',encoding='utf-8') as file:
            lines = file.split('\n')
            print(uri,len(lines))
        """
        file = open(uri,'rt',encoding='utf-8')
        data = file.read()
        file.close()
        lines = data.split('\n')
        # print(uri,len(lines))

        for i in lines:
            if i == '':
                # sentence_index += 1
                pass
            elif (not i.startswith('#')):
                current_document.append(i.split('\t'))
                """
                if current_document[-1][3] == 'PUNCT':
                    # punct_count += 1
                    current_punct_count += 1
                """
        
        #<ENTROPY>
        list_for_entropy = [i[3] for i in current_document]
        entropy_csv = open('entropy_csv.csv','at',encoding='utf-8')
        entropy_csv.write(uri + ';' + str(entropy(list_for_entropy)) + '\n')
        entropy_csv.close()

        #</ENTROPY>

        documents.append(current_document)

        # return current_punct_count

    else:
        print(f'ERROR: {uri} does not exist!')

# del(current_document)

documents_ratios = []

doc_count = 0
docs_list = listdir(directory_path)
len_docs_list = len(docs_list)
for i in docs_list:
    # s = f'Parsing document {doc_count+1}/{len_docs_list}'
    # print(s,end='\b'*len(s))
    current_document = []
    if not i.endswith('.conllu'):
        continue
    """
    parse_conllu(directory_path+i)
    punct_count += current_punct_count
    current_punct_count = 0
    """
    parse_conllu(directory_path+i)
    # punct_count += parse_conllu(directory_path+i)
    documents_ratios.append({
        "name": i
    })
    doc_count += 1

del(parse_conllu)
del(directory_path)

# print('punct_count:',punct_count)
# exit()

# </PARSING>

# print('Documents:',documents[:1])

# <AMBIGUITIES_LIST>

# ambiguities_reference_list = []
ambiguities_file = open('AWL_POS_ambiguous_words.csv','rt',encoding='utf-8')
ambiguities_reference_list = ambiguities_file.read().lower().split('\n')
ambiguities_file.close()
del(ambiguities_file)

# ambiguities_match_count = [0 for i in range(len(ambiguities_reference_list))]
# ambiguities_word_count = ambiguities_match_count[:]




"""
words_match_count = 0
all_words_count = 0
# for i in documents:
for i in range(len(documents)):
    all_tokens_in_document = 0
    all_ambiguities_in_document = 0
    # for j in i:
    for j in range(len(documents[i])):
        # if j[2] in ambiguities_reference_list:
        if documents[i][j][2] in ambiguities_reference_list:
            words_match_count += 1
            all_ambiguities_in_document += 1
        all_words_count += 1
        all_tokens_in_document += 1
    # documents_ratios.append({
    #     "name": 
    # })
    documents_ratios[i]["token_count"] = all_tokens_in_document
    documents_ratios[i]["ambiguity_count"] = all_ambiguities_in_document

print(f'{words_match_count}/{all_words_count}')

csv_str = 'x,y'
csv_str2 = 'x,y'

for i in documents_ratios:
    i["token_ratio"] = i["token_count"]/all_words_count
    i["ambiguity_ratio"] = i["ambiguity_count"]/words_match_count
    i["general_indicator"] = i["ambiguity_ratio"]/i["token_ratio"]

    csv_str += '\n'
    csv_str += i['name']
    csv_str += ','
    csv_str += str(i['general_indicator'])

    csv_str2 += '\n'
    csv_str2 += i['name']
    csv_str2 += ','
    csv_str2 += str(i['token_ratio'])
    csv_str2 += ','
    csv_str2 += str(i['ambiguity_ratio'])

file = open('ambiguities_ratios.csv','wt',encoding='utf-8')
file.write(csv_str)
file.close()

file = open('ambiguities_ratios_plot.csv','wt',encoding='utf-8')
file.write(csv_str2)
file.close()
"""

# plt.plot([i["general_indicator"] for i in documents_ratios])

"""
x_labels = [i["name"] for i in documents_ratios]
y_values = [i["general_indicator"] for i in documents_ratios]

# plt.plot(x_labels,y_values)
plt.bar(x_labels,y_values)
plt.show()
"""

# print(documents_ratios)


# </AMBIGUITIES_LIST>


















# print('Ambiguities:',ambiguities_reference_list[:5])





"""
def cosine_similarity(a,b):
    # return (np.prod(a,b)/np.prod(np.sqrt(np.sum(np.square(a))),np.sqrt(np.sum(np.square(b)))))
    # return (np.prod(a,b)/np.prod(sqrt(np.sum(np.square(a))),sqrt(np.sum(np.square(b)))))
    # return (np.prod(a,b)/(np.sqrt(np.sum(np.square(a)))*np.sqrt(np.sum(np.square(b)))))
    try:
        return (np.dot(a,b)/(sqrt(np.sum(np.square(a)))*sqrt(np.sum(np.square(b)))))
    except:
        return 0


del(cosine_similarity)
"""


# <ML_CLASSES>


"""
# PUTTING THIS IN COMMENT SO THAT IT DOES NOT INTERFERE
saving = True
training = False

file = open('word_embedding.csv','rt',encoding='utf-8')
#length : 18555 (to be verified)
word_embedding_list = file.read().split('\n')
file.close()

word_embedding_list = [a.lower() for a in word_embedding_list]
# word_embedding_list = np.asarray(word_embedding_list,dtype=np.str)
# print(word_embedding_list.index('study'))
# word_embedding_list = np.asarray(word_embedding_list,dtype=str)

len_word_embedding_list = len(word_embedding_list)
"""
# default_embedding = np.zeros(len_word_embedding_list,dtype=np.int8)
# default_embedding = np.zeros(len_word_embedding_list,dtype=np.int32)

class word2vec_vectorizer:
    word2vec_vectors_list = []
    d = 0
    lower_window = 0
    upper_window = 0
    # def __init__(self,uri,lower_window,upper_window,d,preprocessing_uri_to_pkl,binary = False):
    def __init__(self,uri,lower_window,upper_window,d,binary = False):
        if exists(uri):
            # t = time.time()
            t = time()
            # self.word2vec_vectors_list = gensim.models.KeyedVectors.load_word2vec_format(r'frwiki_20180420_300d.txt.bz2', binary=False)
            # self.word2vec_vectors_list = KeyedVectors.load_word2vec_format(r'frwiki_20180420_300d.txt.bz2', binary=False)
            # self.word2vec_vectors_list = KeyedVectors.load_word2vec_format(uri, binary=binary)
            if uri in loaded_semantics_models.keys():
                self.word2vec_vectors_list = loaded_semantics_models[uri]
            else:
                self.word2vec_vectors_list = KeyedVectors.load_word2vec_format(uri, binary=binary)
                loaded_semantics_models[uri] = self.word2vec_vectors_list
                # joblib.dump(self.word2vec_vectors_list,preprocessing_uri_to_pkl)
            # print(f'Time for loading {uri}: {time.time()-t}s')
            print(f'Time for loading {uri}: {time()-t}s')
            self.d = d

            self.lower_window = lower_window
            self.upper_window = upper_window
        else:
            print(f'[ERROR] File {uri} does not exist!')
    def fit(self,a,b):
        # pass
        return self
    def transform(self,input_x):
        # print(f'Transforming {input_x}...')
        # transform_result = np.zeros((len(input_x),self.d),dtype=np.int32)
        # transform_result = np.zeros((len(input_x),self.d*(1+self.lower_window+self.upper_window)),dtype=np.int32)
        transform_result = np.zeros((len(input_x),self.d*(1+self.lower_window+self.upper_window)),dtype=np.float32)
        for i in range(len(input_x)):
            # result = np.zeros(len_word_embedding_list,dtype=np.int32)
            for j in range(len(input_x[i])):
                try:
                    # result[word_embedding_list.index(input_x.lower())] = 1
                    # transform_result[i] = self.word2vec_vectors_list[input_x[i]]
                    # transform_result[i][self.d*j:self.d*(j+1)] = self.word2vec_vectors_list[input_x[i]]
                    transform_result[i][self.d*j:self.d*(j+1)] = self.word2vec_vectors_list[input_x[i][j].lower()]
                except:
                    pass
            # print(index)
            # return result
        return transform_result

    





class hash_lexicon_vectorizer:
    # word2vec_vectors_list = []
    d = 0
    lower_window = 0
    upper_window = 0
    fh = None
    # def __init__(self,uri,lower_window,upper_window,d,preprocessing_uri_to_pkl,binary = False):
    def __init__(self,lower_window,upper_window,d):
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.d = d
        self.fh = FeatureHasher(n_features=self.d)
    def fit(self,a,b):
        # pass
        return self
    def transform(self,input_x):
        
        # print(f'Transforming {input_x}...')
        # transform_result = np.zeros((len(input_x),self.d),dtype=np.int32)
        # transform_result = np.zeros((len(input_x),self.d*(1+self.lower_window+self.upper_window)),dtype=np.int32)
        # transform_result = np.zeros((len(input_x),64*(1+self.lower_window+self.upper_window)),dtype=np.float32)
        # transform_result = np.zeros((len(input_x),self.d*(1+self.lower_window+self.upper_window)),dtype=np.int8)
        transform_result = np.zeros((len(input_x),self.d),dtype=np.int8)
        for i in range(len(input_x)):
            # result = np.zeros(len_word_embedding_list,dtype=np.int32)
            local_dict = {}
            # CETTE PARTIE RESTE A VERIFIER
            for j in range(len(input_x[i])):
                if j == self.lower_window:
                    local_dict[input_x[i][j]] = np.int8(1)
                    # local_dict[input_x[i][j]] = 1
                    # local_dict[input_x[i][j]] = -10
                    # local_dict[input_x[i][j]] = np.int8(100)
                else:
                    # local_dict[input_x[i][j]] = np.int8(-1)
                    # local_dict[input_x[i][j]] = np.int8(j-self.lower_window)
                    # local_dict[input_x[i][j]] = j-self.lower_window
                    """
                    if j < self.lower_window:
                        local_dict[input_x[i][j]] = -(j+1)
                    else:
                        local_dict[input_x[i][j]] = -(len(input_x[i])-j+1)
                    """
                    local_dict[input_x[i][j]] = -abs(j-self.lower_window)
            # transform_result[i] = self.fh.transform([local_dict]).toarray()[0]
            transform_result[i] = np.int8(self.fh.transform([local_dict]).toarray()[0]) # BUT WILL IT MAKE A DIFFERENCE?
            """
            if self.lower_window == 0:
                for j in transform_result[i]:
                    if j < 0:
                        print(j)
            """
            """
            for j in range(len(input_x[i])):
                try:
                    # result[word_embedding_list.index(input_x.lower())] = 1
                    # transform_result[i] = self.word2vec_vectors_list[input_x[i]]
                    # transform_result[i][self.d*j:self.d*(j+1)] = self.word2vec_vectors_list[input_x[i]]
                    # transform_result[i][self.d*j:self.d*(j+1)] = [np.int8(k) for k in np.binary_repr(abs(hash(input_x[i][j].lower())))]
                    transform_result[i][self.d*j:self.d*(j+1)] = self.fh.transform()
                except:
                    pass
            """
            # print(index)
            # return result
        return transform_result











class word_embedding:
    word_embedding_list = []
    lower_window = 0
    upper_window = 0
    def __init__(self,uri,lower_window,upper_window):
        if exists(uri):
            # file = open('word_embedding.csv','rt',encoding='utf-8')
            file = open(uri,'rt',encoding='utf-8')
            #length : 18555 (to be verified)
            self.word_embedding_list = file.read().split('\n')
            file.close()

            self.word_embedding_list = [a.lower() for a in self.word_embedding_list]
            # print(self.word_embedding_list)
            # word_embedding_list = np.asarray(word_embedding_list,dtype=np.str)
            # print(word_embedding_list.index('study'))
            # word_embedding_list = np.asarray(word_embedding_list,dtype=str)

            # self.len_word_embedding_list = len(word_embedding_list)
            self.len_word_embedding_list = len(self.word_embedding_list)

            self.lower_window = lower_window
            self.upper_window = upper_window
        else:
            print(f'[ERROR] File {uri} does not exist!')
    def fit(self,a,b):
        # pass
        return self
    def transform(self,input_x):
        # print(f'Transforming {input_x}...')
        # print(input_x)
        # transform_result = np.zeros((len(input_x),self.len_word_embedding_list),dtype=np.int32)
        # transform_result = np.zeros((len(input_x),self.len_word_embedding_list*(1+self.lower_window+self.upper_window)),dtype=np.int32)
        transform_result = np.zeros((len(input_x),self.len_word_embedding_list*(1+self.lower_window+self.upper_window)),dtype=np.int8)
        for i in range(len(input_x)):
            # result = np.zeros(len_word_embedding_list,dtype=np.int32)
            for j in range(len(input_x[i])):
                try:
                    # result[word_embedding_list.index(input_x.lower())] = 1
                    # transform_result[i][word_embedding_list.index(input_x[i].lower())+(j*self.len_word_embedding_list)] = 1
                    # transform_result[i][self.word_embedding_list.index(input_x[i].lower())+(j*self.len_word_embedding_list)] = 1
                    transform_result[i][self.word_embedding_list.index(input_x[i][j].lower())+(j*self.len_word_embedding_list)] = 1 #OK
                except:
                    pass
            # print(index)
            # return result
        return transform_result



# </ML_CLASSES>


# WE = word_embedding('test_word_embedding.csv',1,1)
# print(WE.transform([['a','a','a'],['a','b','c'],['d','d','d']]))

# exit()

# <SYSTEMS>

training_size_cap = 200000
# training_size_cap = 1000

systems_parameters = []



preprocessing_parameters = [
    [True,4096],
    [True,4096*3],
    [False,100],
    [False,300]
]

window_parameters = [
    [0,0],
    [1,1],
    [2,2],
    [3,3],
    [1,0],
    [2,0],
    [3,0],
    [0,1],
    [0,2],
    [0,3]
]

core_parameters = [
    ['DTC'],
    ['NearCent'],
    ['L-SVC'],
    ['MLP']
]

overwrite_PRFS_summary = True

for i in preprocessing_parameters:
    s = ''
    if i[0]:
        s += 'hashFH-'
    else:
        s += 'w2v-'
    s += str(i[1]) + '_'
    for j in window_parameters:
        # t = s[:]
        t = ''
        t += str(j[0]) + '-' + str(j[1]) + '_'
        for k in core_parameters:
            u = t[:]
            # u += str(k[0])
            u += str(k[0]) + '_' + str(i[1])

            systems_parameters.append({
                "lexicon": i[0],
                "hashing": True,
                "algorithm": k[0],
                "preprocessing_uri": 'enwiki_20180420_'+ str(i[1]) +'d.txt.bz2',
                # "preprocessing_uri_to_pkl": 'en_100d.pkl',
                "preprocessing_uri_to_pkl": s + '.pkl',
                "main_processing_uri_to_pkl": u + '.pkl',
                "model_uri": s + u + '.pkl',
                "lower_window": j[0],
                "upper_window": j[1],
                "tol": 1e-3,
                "solver": 'adam',
                # "d": 512,
                "d": i[1],
                "binary": False,
                # "allow_training": True,
                "allow_training": False,
                # "allow_saving": True,
                "allow_saving": False,
                "allow_global_loading": False,
                "allow_preprocessing_loading": False,
                "allow_main_processing_loading": True,
                "no_training_if_main_processing_to_uri_pkl_exists": True
            })

del(preprocessing_parameters)
del(core_parameters)
del(window_parameters)

systems = []
loaded_semantics_models = {}

labels = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','SCONJ','VERB','X']

# PRFS_table = [['preprocessing','main_processing','not_pos/pos','train/test','precision','recall','f1score','support']]
# PRFS_table = [['preprocessing','main_processing','not_pos/pos','train/test','data_type',*['POS' + str(i) for i in range(1,17)]]]
# PRFS_table = [['preprocessing','main_processing','not_pos/pos','train/test','data_type',*['POS' + str(i) for i in range(1,18)]]] # CHANGED TO 18 TO REACH 17 (INCLUDED)
# PRFS_table = [['preprocessing','main_processing','not_pos/pos','train/test','data_type',*['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','SCONJ','VERB','X']] # CHANGED TO 18 TO REACH 17 (INCLUDED)
PRFS_table = [['preprocessing','main_processing','not_pos/pos','train/test','data_type',*labels]] # CHANGED TO 18 TO REACH 17 (INCLUDED)
if overwrite_PRFS_summary:
    file = open('PRFS_summary.csv','wt',encoding='utf-8')
    file.write(';'.join(PRFS_table[0]) + '\n')
    # file.write('\t'.join(PRFS_table[0]) + '\n')
    file.close()
# PRFS_table.pop(0)
del(PRFS_table[0])

# for i in systems_parameters:
for i in range(len(systems_parameters)):
    # if exists(systems_parameters[i]["model_uri"]) and systems_parameters[i]["allow_global_loading"]:
    if systems_parameters[i]["allow_global_loading"] and exists(systems_parameters[i]["model_uri"]):
        systems.append(joblib.load(systems_parameters[i]["model_uri"]))
        print(f'Loaded {systems_parameters[i]["model_uri"]}')
    else:
        preprocessing_obj = None
        main_processing_obj = None

        if systems_parameters[i]['allow_preprocessing_loading'] and exists(systems_parameters[i]['preprocessing_uri_to_pkl']):
            # preprocessing_obj = ('preprocessing',joblib.load(systems_parameters[i]['preprocessing_uri_to_pkl']))
            preprocessing_obj = joblib.load(systems_parameters[i]['preprocessing_uri_to_pkl'])
        else:
            if systems_parameters[i]['lexicon']:
                if systems_parameters[i]['hashing']:
                    preprocessing_obj = ('hashing_lexicon',hash_lexicon_vectorizer(
                        lower_window=systems_parameters[i]["lower_window"],
                        upper_window=systems_parameters[i]["upper_window"],
                        d=systems_parameters[i]["d"], #ADDING THIS FOR FH
                    ))
                else:
                    preprocessing_obj = ('word_embedding',word_embedding(
                        systems_parameters[i]["preprocessing_uri"],
                        lower_window=systems_parameters[i]["lower_window"],
                        upper_window=systems_parameters[i]["upper_window"]
                    ))
            else:
                preprocessing_obj = ('word2vec_vectorizer',word2vec_vectorizer(
                    systems_parameters[i]["preprocessing_uri"],
                    lower_window=systems_parameters[i]["lower_window"],
                    upper_window=systems_parameters[i]["upper_window"],
                    d=systems_parameters[i]["d"],
                    binary=systems_parameters[i]["binary"]
                ))

        if systems_parameters[i]['allow_main_processing_loading'] and exists(systems_parameters[i]['main_processing_uri_to_pkl']):
            # preprocessing_obj = ('main_processing',joblib.load(systems_parameters[i]['main_processing_uri_to_pkl']))
            # main_processing_obj = ('main_processing',joblib.load(systems_parameters[i]['main_processing_uri_to_pkl']))
            main_processing_obj = joblib.load(systems_parameters[i]['main_processing_uri_to_pkl'])
        else:
            if systems_parameters[i]['algorithm'] == 'MLP':
                main_processing_obj = ('mlp',MLPClassifier(
                    hidden_layer_sizes=(64,32),
                    solver=systems_parameters[i]["solver"],
                    activation='relu',
                    max_iter = 200,
                    tol=systems_parameters[i]["tol"],
                    random_state = 1,
                    early_stopping=True
                ))
            elif systems_parameters[i]['algorithm'] == 'L-SVC':
                main_processing_obj = ('l-svc',LinearSVC(
                    # hidden_layer_sizes=(64,32),
                    # solver=systems_parameters[i]["solver"],
                    # activation='relu',
                    max_iter = 1000,
                    tol=systems_parameters[i]["tol"],
                    random_state = 1,
                    # early_stopping=True,
                    # verbose = 1,
                    verbose = 0,
                    dual=False
                ))
            elif systems_parameters[i]['algorithm'] == 'NearCent':
                main_processing_obj = ('nearcent',NearestCentroid(
                    # n_neighbors = 10,
                    # n_neighbors = 5,
                    # metric = cosine_similarity
                    # metric = 'manhattan'
                    metric = 'euclidean'
                ))
            
            elif systems_parameters[i]['algorithm'] == 'CNB':
                main_processing_obj = ('cnb',ComplementNB(
                    
                ))
            
            elif systems_parameters[i]['algorithm'] == 'DTC':
                main_processing_obj = ('dtc',DecisionTreeClassifier(
                    # max_depth = 256
                    max_depth = 32
                    # max_depth = 64
                    # max_depth = 128
                    # max_depth = 16
                ))
        try:
            systems.append(
                Pipeline(
                    [
                        preprocessing_obj,
                        main_processing_obj
                    ],
                    # verbose=True
                    verbose=False
                )
            )
        except:
            print('ERROR, could not create Pipeline. Systems parameters:',systems_parameters[i], preprocessing_obj, main_processing_obj)
            continue

    # </SYSTEMS>



    # print('Systems:',systems[:5])

    # pos_ambiguities_count = 0


    # <FITTING>


    #for i in range(len(systems)):
    print(f'Preparing systems[{i}]...')
    __X = []
    __y = []
    X_train = []
    y_train = []
    pos_ambiguities_X = []
    pos_ambiguities_y = []
    not_pos_ambiguities_X = []
    not_pos_ambiguities_y = []
    for j in range(len(documents)):
        # m = systems[i]["lower_window"]
        # maximum = systems[i]["upper_window"]
        # m = systems_parameters[i].lower_window
        # maximum = systems_parameters[i].upper_window
        m = -systems_parameters[i]["lower_window"]
        maximum = systems_parameters[i]["upper_window"]
        len_documents_j = len(documents[j])
        # for k in range(len(documents[j])):
        for k in range(len_documents_j):
            # if documents[j][k][3] != 'PUNCT' and ((not pos_ambiguities_only) or documents[j][k][2] in ambiguities_reference_list):
            if documents[j][k][3] != 'PUNCT' and ((not pos_ambiguities_only) or documents[j][k][1].lower() in ambiguities_reference_list):
                n = m
                local_list = []
                while n <= maximum:
                    if n < 0 or n >= len_documents_j:
                        local_list.append('')
                    else:
                        # local_list.append(documents[j][n])
                        local_list.append(documents[j][n][2])
                    n += 1
                __X.append(local_list)
                X_train.append(local_list)
                __y.append(documents[j][k][3])
                y_train.append(documents[j][k][3])
                if documents[j][k][1].lower() in ambiguities_reference_list:
                    # pos_ambiguities_count += 1
                    pos_ambiguities_X.append(local_list)
                    pos_ambiguities_y.append(documents[j][k][3])
                else:
                    # pos_ambiguities_count += 1
                    not_pos_ambiguities_X.append(local_list)
                    not_pos_ambiguities_y.append(documents[j][k][3])
            m += 1
            maximum += 1
            if len(X_train) > training_size_cap:
                break
        if len(X_train) > training_size_cap:
            break
    
    pos_X_train, pos_X_test, pos_y_train, pos_y_test = train_test_split(pos_ambiguities_X,pos_ambiguities_y,random_state=0)
    not_pos_X_train, not_pos_X_test, not_pos_y_train, not_pos_y_test = train_test_split(not_pos_ambiguities_X,not_pos_ambiguities_y,random_state=0)
    
    # X_train = X_train[:training_size_cap]
    # y_train = y_train[:training_size_cap]
    
    X_train = pos_X_train[:]
    X_train[len(X_train):] = not_pos_X_train
    X_train = X_train[:training_size_cap] # ?
    y_train = pos_y_train[:]
    y_train[len(y_train):] = not_pos_y_train
    y_train = y_train[:training_size_cap] # ?

    # print('X_train:',X_train[:5])
    # print('y_train:',y_train[:5])

    # print(pos_ambiguities_count,len(pos_ambiguities_X_train),len(pos_ambiguities_y_train))
    # print(pos_ambiguities_X_train[:10])
    # print(pos_ambiguities_y_train[:10])
    # assert pos_ambiguities_count == len(pos_ambiguities_X_train) and pos_ambiguities_count == len(pos_ambiguities_y_train)
    # print(len(X_train),len(y_train))
    # assert pos_ambiguities_count == len(pos_ambiguities_X) and pos_ambiguities_count == len(pos_ambiguities_y)
    assert len(pos_ambiguities_X) == len(pos_ambiguities_y)
    assert len(X_train) == len(y_train)

    print(f'Size of training sample (global): {len(X_train)}')
    # print(f'Number of POS ambiguities: {pos_ambiguities_count}')
    print(f'Number of POS ambiguities (in training set): {len(pos_X_train)}')
    
    # print(training)
    # if training:
    if systems_parameters[i]["allow_training"] and ((not exists(systems_parameters[i]["main_processing_uri_to_pkl"])) or (not systems_parameters[i]["no_training_if_main_processing_to_uri_pkl_exists"])):
        print(f'Fitting systems[{i}]...')
        systems[i].fit(X_train,y_train)
        # IT SEEMS THAT OTHER PARAMETERS EXIST
        # train_sizes, train_scores, valid_scores = learning_curve(
        #     systems[i], X_train, y_train
        # )
        # plt.plot(train_scores)
    
    
    if systems_parameters[i]["allow_saving"]:
        if systems_parameters[i]['allow_global_loading']:
            joblib.dump(systems[i],systems_parameters[i]["model_uri"])
            print(f'Saved file: {systems_parameters[i]["model_uri"]}')
        else:
            if systems_parameters[i]['allow_preprocessing_loading']:
                joblib.dump(preprocessing_obj,systems_parameters[i]['preprocessing_uri_to_pkl'])
                print(f'Saved file: {systems_parameters[i]["preprocessing_uri_to_pkl"]}')
            # if systems_parameters[i]['allow_main_processing_loading']:
            if systems_parameters[i]['allow_main_processing_loading'] and ((not exists(systems_parameters[i]["main_processing_uri_to_pkl"])) or (not systems_parameters[i]["no_training_if_main_processing_to_uri_pkl_exists"])):
                joblib.dump(main_processing_obj,systems_parameters[i]['main_processing_uri_to_pkl'])
                print(f'Saved file: {systems_parameters[i]["main_processing_uri_to_pkl"]}')
    
    not_pos_y_train_pred = systems[i].predict(not_pos_X_train)
    not_pos_y_test_pred = systems[i].predict(not_pos_X_test)
    pos_y_train_pred = systems[i].predict(pos_X_train)
    pos_y_test_pred = systems[i].predict(pos_X_test)

    PRFS_not_pos_y_train = precision_recall_fscore_support(not_pos_y_train,not_pos_y_train_pred,zero_division=0,labels=labels)
    PRFS_not_pos_y_test = precision_recall_fscore_support(not_pos_y_test,not_pos_y_test_pred,zero_division=0,labels=labels)
    PRFS_pos_y_train = precision_recall_fscore_support(pos_y_train,pos_y_train_pred,zero_division=0,labels=labels)
    PRFS_pos_y_test = precision_recall_fscore_support(pos_y_test,pos_y_test_pred,zero_division=0,labels=labels)

    """
    PRFS_table.append([
        systems_parameters[i]['preprocessing_uri_to_pkl'],
        systems_parameters[i]['main_processing_uri_to_pkl'],
        'not_pos',
        'train',
        str(PRFS_not_pos_y_train[0]),
        str(PRFS_not_pos_y_train[1]),
        str(PRFS_not_pos_y_train[2]),
        str(PRFS_not_pos_y_train[3])
    ])
    PRFS_table.append([
        systems_parameters[i]['preprocessing_uri_to_pkl'],
        systems_parameters[i]['main_processing_uri_to_pkl'],
        'not_pos',
        'test',
        str(PRFS_not_pos_y_test[0]),
        str(PRFS_not_pos_y_test[1]),
        str(PRFS_not_pos_y_test[2]),
        str(PRFS_not_pos_y_test[3])
    ])
    PRFS_table.append([
        systems_parameters[i]['preprocessing_uri_to_pkl'],
        systems_parameters[i]['main_processing_uri_to_pkl'],
        'pos',
        'train',
        str(PRFS_pos_y_train[0]),
        str(PRFS_pos_y_train[1]),
        str(PRFS_pos_y_train[2]),
        str(PRFS_pos_y_train[3])
    ])
    PRFS_table.append([
        systems_parameters[i]['preprocessing_uri_to_pkl'],
        systems_parameters[i]['main_processing_uri_to_pkl'],
        'pos',
        'test',
        str(PRFS_pos_y_test[0]),
        str(PRFS_pos_y_test[1]),
        str(PRFS_pos_y_test[2]),
        str(PRFS_pos_y_test[3])
    ])
    """

    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','train','precision',*[str(j) for j in PRFS_not_pos_y_train[0]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','train','recall',*[str(j) for j in PRFS_not_pos_y_train[1]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','train','f1score',*[str(j) for j in PRFS_not_pos_y_train[2]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','train','support',*[str(j) for j in PRFS_not_pos_y_train[3]]])
    
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','test','precision',*[str(j) for j in PRFS_not_pos_y_test[0]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','test','recall',*[str(j) for j in PRFS_not_pos_y_test[1]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','test','f1score',*[str(j) for j in PRFS_not_pos_y_test[2]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'not_pos','test','support',*[str(j) for j in PRFS_not_pos_y_test[3]]])
    
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','train','precision',*[str(j) for j in PRFS_pos_y_train[0]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','train','recall',*[str(j) for j in PRFS_pos_y_train[1]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','train','f1score',*[str(j) for j in PRFS_pos_y_train[2]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','train','support',*[str(j) for j in PRFS_pos_y_train[3]]])
    
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','test','precision',*[str(j) for j in PRFS_pos_y_test[0]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','test','recall',*[str(j) for j in PRFS_pos_y_test[1]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','test','f1score',*[str(j) for j in PRFS_pos_y_test[2]]])
    PRFS_table.append([systems_parameters[i]['preprocessing_uri_to_pkl'],systems_parameters[i]['main_processing_uri_to_pkl'],'pos','test','support',*[str(j) for j in PRFS_pos_y_test[3]]])





    #THIS SHOULD REDUCE MEMORY USAGE
    file = open('PRFS_summary.csv','at',encoding='utf-8')
    while len(PRFS_table) > 0:
        file.write(';'.join(PRFS_table[0]) + '\n')
        # PRFS_table.pop(0)
        del(PRFS_table[0])
    file.close()
    del(file)


    
    

    # print('not_pos, train:',PRFS_not_pos_y_train)
    # print('not_pos, test:',PRFS_not_pos_y_test)
    # print('pos, train:',PRFS_pos_y_train)
    # print('pos, test:',PRFS_pos_y_test)

    # precision_recall_fscore_support

    if display:

        

        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(6,6),sharey='all',sharex='all',squeeze=True)
        # fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(9,6),sharey='row',sharex='col')
        # fig, axes = plt.subplots(nrows=2,ncols=2)
        
        # empty_labels = ['','','','','','','','','','','','','','','','','']
        disp = ConfusionMatrixDisplay.from_predictions(not_pos_y_train,not_pos_y_train_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[0],colorbar=False)
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('Vraie classe (sans ambiguïté)')
        # ConfusionMatrixDisplay.from_predictions(not_pos_y_train,not_pos_y_train_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[0])
        disp = ConfusionMatrixDisplay.from_predictions(not_pos_y_test,not_pos_y_test_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[1],colorbar=False)
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        # ConfusionMatrixDisplay.from_predictions(not_pos_y_test,not_pos_y_test_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[1])
        # ConfusionMatrixDisplay.from_predictions(not_pos_y_train+not_pos_y_test,not_pos_y_train_pred+not_pos_y_test_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[2])
        # ConfusionMatrixDisplay.from_predictions([*not_pos_y_train,*not_pos_y_test],[*not_pos_y_train_pred,*not_pos_y_test_pred],normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[2],colorbar=False)
        # ConfusionMatrixDisplay.from_predictions([*not_pos_y_train,*not_pos_y_test],[*not_pos_y_train_pred,*not_pos_y_test_pred],normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[2])
        # ConfusionMatrixDisplay.from_predictions(pos_y_train,pos_y_train_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[3],colorbar=False)
        disp = ConfusionMatrixDisplay.from_predictions(pos_y_train,pos_y_train_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[2],colorbar=False)
        disp.ax_.set_xlabel('Prédiction (données d\'entrainement)')
        disp.ax_.set_ylabel('Vraie classe (avec ambiguïté)')
        # ConfusionMatrixDisplay.from_predictions(pos_y_train,pos_y_train_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[3])
        # ConfusionMatrixDisplay.from_predictions(pos_y_test,pos_y_test_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[4],colorbar=False)
        disp = ConfusionMatrixDisplay.from_predictions(pos_y_test,pos_y_test_pred,normalize='true',cmap='Blues',include_values=False,xticks_rotation='vertical',ax=axes.flatten()[3],colorbar=False)
        disp.ax_.set_xlabel('Prédiction (données de test)')
        disp.ax_.set_ylabel('')
        
        
        plt.tight_layout()
        plt.show()

        

    
    


    # UPDATE CSV

    if not exists('systems_summary.csv'):
        document_type_regex = re.compile('^[^_]*_[^_]*')
        table = []
        table.append(['index','document','document_type','is_POS_ambiguity','token','lemma','POS'])
        token_index = 0
        # file = open('systems_summary.csv','xt',encoding='utf-8')
        file = open('systems_summary.csv','wt',encoding='utf-8')
        for j in range(len(documents)):
            document_type = document_type_regex.findall(docs_list[j])[0]
            for k in range(len(documents[j])):
                local_list = [
                    str(token_index),
                    docs_list[j],
                    # document_type_regex.findall(docs_list[j])[0],
                    document_type,
                    str(int(documents[j][k][1].lower() in ambiguities_reference_list)),
                    documents[j][k][1],
                    documents[j][k][2],
                    documents[j][k][3]
                ]

                table.append(local_list)

                token_index += 1
        # csv_str = ''
        # table = [';'.join(j) for j in table]
        table = ['\t'.join(j) for j in table]
        file.write('\n'.join(table))
        file.close()
        print('Created systems_summary.csv')
    file = open('systems_summary.csv','rt',encoding='utf-8')
    lines = file.read().split('\n')
    file.close()
    # table = [j.split(';') for j in lines]
    table = [j.split('\t') for j in lines]
    del(lines)

    # index_of_current_system = table[0].index(systems_parameters[i]["model_uri"])
    index_of_current_system = -1

    # if index_of_current_system == -1:
    if systems_parameters[i]["model_uri"] not in table[0]:
        table[0].append(systems_parameters[i]["model_uri"])
        index_of_current_system = len(table[0])-1
        for j in range(1,len(table)):
            table[j].append('')
    else:
        index_of_current_system = table[0].index(systems_parameters[i]["model_uri"])
    
    print(f'index_of_current_system: {index_of_current_system}')
    
    # X_train_predict = [*not_pos_y_train_pred,*not_pos_y_test_pred,*pos_y_train_pred,*pos_y_test_pred] # TESTING
    # X_train_predict = systems[i].predict(X_train) # TESTING
    X_train_predict = systems[i].predict(__X) # TESTING
    print(len(X_train_predict))
    assert len(X_train_predict) == len(__X)
    index_difference = 0
    len_table = len(table)
    for j in range(len(X_train_predict)):
        # [4] -> lemma ?
        # [5] -> lemma ? (due to shift after adding is_POS_ambiguity)
        # while table[j+index_difference+1][4] != X_train[systems_parameters[i]["lower_window"]] and index_difference < len(table):
        # while (j+index_difference+1) < len(table) and table[j+index_difference+1][4] != X_train[systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][4] != X_train[systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][5] != X_train[systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][5] != X_train[j][systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][5] != X_train_predict[j][systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][5] == 'PUNCT':
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][6] == 'PUNCT':
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][5] != X_train_predict[j][systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][4] != X_train_predict[j][systems_parameters[i]["lower_window"]]:
        # while (j+index_difference+1) < len_table and table[j+index_difference+1][6] == 'PUNCT':
        while (j+index_difference+1) < len_table and (len(table[j+index_difference+1]) < 7 or table[j+index_difference+1][6] == 'PUNCT'):
            index_difference += 1
        # while (j+index_difference+1) < len_table:
        #     try:
        #         if table[j+index_difference+1][4] != X_train_predict[j][systems_parameters[i]["lower_window"]]:
        #             index_difference += 1
        #         else:
        #             break
        #     except:
        # #         break
        # while table[j+index_difference+1][6] == 'PUNCT':
        #     index_difference += 1
            
        try:
            table[j+index_difference+1][index_of_current_system] = X_train_predict[j]
        except:
            pass
    
    # table = [';'.join(j) for j in table]
    table = ['\t'.join(j) for j in table]
    file = open('systems_summary.csv','wt',encoding='utf-8')
    file.write('\n'.join(table))
    file.close()
    print('Updated systems_summary.csv')




















    # UNLOADING TO SAVE MEMORY SPACE

    systems[i] = None 


























    print('='*32)

# plt.show()

# </FITTING>

"""
PRFS_table = [';'.join(i) for i in PRFS_table]
PRFS_str = '\n'.join(PRFS_table)

file = open('PRFS_summary.csv','wt',encoding='utf-8')
file.write(PRFS_str)
file.close()
"""

# print(f'Complete runtime: {time.time()-global_t}s')
print(f'Complete runtime: {time()-global_t}s')