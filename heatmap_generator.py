from os.path import exists

file = open('CSV_for_heatmap.csv','rt',encoding='utf-8')
data = file.read()
file.close()

table = data.split('\n')
table = [i.split('\t') for i in table]

dico = {}
last_word = table[0][0]
last_word_index = 0

possible_values = ['a','c','d','e','i','j','m','n','p','r','t','u','v','x']
replacement = ['ART','CONJ','DET','e','i','ADJ','NUM','NOM','PRO','ADV','t','INTERJ','VER','NEG']

for i in possible_values:
    dico[i] = {
        "total_count": 0
    }
    for j in possible_values:
        dico[i][j] = 0

# print(dico) # OK

for i in range(len(table)):

    if table[i][0] != last_word:
        """
        for j in range(last_word_index,i):
            if table[j][1] not in dico.keys():
                dico[table[j][i]] = {}
                for k in range(j,i):
                    # if j == k:
                    #     continue
                    if table[k][1] not in dico[table[j][i]].keys():
                        dico[table[j][i]][table[k][1]]
        """

        list_of_POS = [table[j][1] for j in range(last_word_index,i)]

        # print(table[last_word_index:i]) # OK

        for j in list_of_POS:
            dico[j]['total_count'] += 1
            # for k in dico.keys():
            #     if k == 'total_count':
            #         continue
            for k in list_of_POS:
                dico[j][k] += 1

        last_word = table[i][0]
        last_word_index = i

list_of_POS = [table[j][1] for j in range(last_word_index,len(table))]

for j in list_of_POS:
    dico[j]['total_count'] += 1
    # for k in dico.keys():
    for k in list_of_POS:
        dico[j][k] += 1

# print(dico['a'])

final_table = [[dico[i][j]/dico[i]['total_count'] for j in dico[i].keys() if j != 'total_count'] for i in dico.keys()]

# print(final_table)
# for i in final_table:
#     print(i)

x_count = 0
y_count = 0

output_str = ''
for i in dico.keys():
    output_str += '\\node[rotate=90] at ('+ str(x_count+0.5) +',2.0){'+ i +'};\n'
    output_str += '\\node at (-2.0,'+ str(-x_count-0.5) +'){'+ i +'};\n' # CHANGED y_count TO x_count
    for j in dico[i].keys():
        if j == 'total_count':
            continue
        # output_str += '\\fill[blue!'+ str(final_table[x_count][y_count]*100) +'] ('+ str(x_count) +','+ str(-y_count) +') rectangle ('+ str(x_count+1) +','+ str(-y_count-1) +');\n'
        y_count += 1
    x_count += 1

for i in range(len(final_table)):
    for j in range(len(final_table[i])):
        # output_str += '\\fill[blue!'+ str(min(final_table[i][j]*100,100)) +'] ('+ str(i) +','+ str(-j) +') rectangle ('+ str(i+1) +','+ str(-j-1) +');\n'
        output_str += '\\fill[blue!'+ str(min(final_table[i][j]*100,100)) +'] ('+ str(j) +','+ str(-i) +') rectangle ('+ str(j+1) +','+ str(-i-1) +');\n'

file_name_counter = 0
file_name = 'heatmap_' + str(file_name_counter) + '.tex'
while exists(file_name):
    file_name_counter += 1
    file_name = 'heatmap_' + str(file_name_counter) + '.tex'

file = open(file_name,'wt',encoding='utf-8')
file.write(output_str)
file.close()
