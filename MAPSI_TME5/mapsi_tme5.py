# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stat
import pydotplus as pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }

def display_BN ( node_names, bn_struct, bn_name, style ):
    graph = pydot.Dot( bn_name, graph_type='digraph')

    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node( name,
                               style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"] )
        graph.add_node( new_node )

    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )

    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )
    plt.show()

##########################################################################################################

# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )

    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )

##########################################################################################################

# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ):
    data = np.loadtxt ( filename, delimiter=',', dtype=np.str ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico

##########################################################################################################


# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z
        size_z *= len ( dico[i] )
        j += 1

    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )

    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices],
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:],
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res

##########################################################################################################

def sufficient_statistics(data , dico , x , y , z):

    res = 0
    table = create_contingency_table(data,dico,x,y,z)
    nb_value_Z = 0

    for value in table:
        Nz , T_xyz = value

        if(Nz != 0):
            nb_value_Z += 1
            nb_line = len(T_xyz)
            nb_col = len(T_xyz[0])

            for i in range(nb_line):
                for j in range(nb_col):

                    Nxyz = T_xyz[i][j]
                    NXcondZ = sum(T_xyz[i])
                    NYcondZ = sum(T_xyz[k][j] for k in range(nb_line))

                    if(NXcondZ != 0 and NYcondZ != 0):
                        res += ((Nxyz - ((NXcondZ*NYcondZ)/Nz))*(Nxyz - ((NXcondZ*NYcondZ)/Nz))/((NXcondZ*NYcondZ)/Nz))

    nb_value_X = len(dico[x])
    nb_value_Y = len(dico[y])

    DoF = (nb_value_X - 1)*(nb_value_Y - 1)*nb_value_Z

    return res,DoF

##########################################################################################################

def indep_score(data,dico,x,y,z):
    nb_value_X = len(dico[x])
    nb_value_Y = len(dico[y])
    chi2,Dof = sufficient_statistics(data,dico,x,y,z)
    nb_value_Z = Dof/((nb_value_X - 1)*(nb_value_Y - 1))
    Dmin = 5*nb_value_X*nb_value_Y*nb_value_Z

    if(len(data[0]) < Dmin):
        return -1,1

    return stat.chi2.sf(chi2,Dof),Dof

##########################################################################################################

def best_candidate(data,dico,x,z,alpha):
    if(x != 0):
        p_values = []
        for y in range(x):
            stats = indep_score(data,dico,x,y,z)
            p_values.append(stats[0])

        if min(p_values) < alpha:
            return [np.argmin(p_values)]

    return []

##########################################################################################################

def create_parents(data,dico,x,alpha):
    z = []
    res = best_candidate(data,dico,x,z,alpha)
    while len(res) != 0:
        z.append(res[0])
        res = best_candidate(data,dico,x,z,alpha)

    return z

##########################################################################################################

def learn_BN_structure(data,dico,alpha):
    res = []
    for x in range(len(data)):
        parents = create_parents(data,dico,x,alpha)
        res.append(parents)

    return res


##########################################################################################################

# names : tableau contenant les noms des variables aléatoires
# data  : tableau 2D contenant les instanciations des variables aléatoires
# dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
names, data, dico = read_csv ( "2015_tme5_asia.csv" )


print("res 1 : ", sufficient_statistics(data, dico, 1, 2, [3]))
print("res 2 : ", sufficient_statistics(data, dico, 0, 1, [2,3]))
print("res 3 : ", sufficient_statistics(data, dico, 1, 3, [2]))
print("res 4 : ", sufficient_statistics(data, dico, 5, 2, [1,3,6]))
print("res 5 : ", sufficient_statistics(data, dico, 0, 7, [4,5]))
print("res 6 : ", sufficient_statistics(data, dico, 2, 3, [5]))

print("indep_score 1 : ",indep_score(data,dico,1,3,[]))
print("indep_score 2 : ",indep_score(data,dico,1,7,[]))
print("indep_score 3 : ",indep_score(data,dico,0,1,[2,3]))
print("indep_score 4 : ",indep_score(data,dico,1,2,[3,4]))

print("best_candidate 1 : ",best_candidate(data,dico,1,[],0.05))
print("best_candidate 2 : ",best_candidate(data,dico,4,[],0.05))
print("best_candidate 3 : ",best_candidate(data,dico,4,[1],0.05))
print("best_candidate 4 : ",best_candidate(data,dico,5,[],0.05))
print("best_candidate 5 : ",best_candidate(data,dico,5,[6],0.05))
print("best_candidate 6 : ",best_candidate(data,dico,5,[6,7],0.05))

print("create_parents 1 : ",create_parents(data,dico,1,0.05))
print("create_parents 2 : ",create_parents(data,dico,4,0.05))
print("create_parents 3 : ",create_parents(data,dico,5,0.05))
print("create_parents 4 : ",create_parents(data,dico,6,0.05))

BN_structure = learn_BN_structure(data,dico,0.05)
print("learn_BN_structure : ",BN_structure)
display_BN(names,BN_structure,"Bayesian Network" , style)




