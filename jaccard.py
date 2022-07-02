################################# <jaccard> ###################################



###############################################################################
#                                                                             #
#   Cette fonction a pour but de calculer l'indice de jaccard entre           #
#   input_a et input_b.                                                       #
#   Il en résulte un score entre 0 et 1.                                      #
#                                                                             #
###############################################################################



def jaccard(input_a,input_b):
    #@1
    #/////////////////// <creation_dictionnaire_compteur> ////////////////////#
    dictionnaire_local = {}
    for a in input_a:
        dictionnaire_local[a] = 1
    for b in input_b:
        if b in dictionnaire_local:
            dictionnaire_local[b] = 2
    for b in input_b:
        if b not in dictionnaire_local:
            dictionnaire_local[b] = 1
    #//////////////////// </creation_dictionnaire_compteur> //////////////////#


    #@2
    #//////////////// <generation_numerateur_et_denominateur> ////////////////#
    c = 0
    d = 0
    for e in dictionnaire_local:
        if dictionnaire_local[e] == 2:
            c += 1
        d += 1
    #//////////////// </generation_numerateur_et_denominateur> ///////////////#

    
    #@3
    #/////////////////////////// <return_resultat> ///////////////////////////#
    return c/max(1,d)
    #/////////////////////////// </return_resultat> //////////////////////////#

#print(jaccard("hello", "john"))

################################# </jaccard> ##################################