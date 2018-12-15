# -*- coding: utf-8 -*-

import numpy as np
from math import *
from pylab import *
import matplotlib.animation as animation

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        if(len(ligne) == 0):
            continue
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

################################################################################################
    
def normale_bidim(X , Z ,PARAM):

    MU_X , MU_Z , SIGMA_X , SIGMA_Z , RO = PARAM
    
    K = 1/(2*math.pi*SIGMA_X*SIGMA_Z*math.sqrt(1-RO*RO))
    
    ALPHA = ((X-MU_X)/SIGMA_X)*(X-MU_X)/SIGMA_X
    
    ALPHA -= (2*RO*(X-MU_X)*(Z-MU_Z))/(SIGMA_X*SIGMA_Z)
    
    ALPHA += ((Z-MU_Z)/SIGMA_Z)*((Z-MU_Z)/SIGMA_Z)
    
    ALPHA *= -1/(2*(1-RO*RO))
    
    return K*math.exp(ALPHA)

################################################################################################

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()
    
################################################################################################
    
def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )

################################################################################################

def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )

################################################################################################

def Q_i (data , current_param , current_weights):
    res = np.zeros((len(data) , 2))
    for i in range(len(data)):
        current_data = data[i]
        Xi = current_data[0]
        Zi = current_data[1]
        
        alpha0 = current_weights[0]*normale_bidim(Xi,Zi,current_param[0])
        alpha1 = current_weights[1]*normale_bidim(Xi,Zi,current_param[1])
        
        Q_Y0 = alpha0/(alpha0+alpha1)
        Q_Y1 = alpha1/(alpha0+alpha1)
        
        res[i][0] = Q_Y0
        res[i][1] = Q_Y1
        
    return res

################################################################################################

def M_step (data , T , current_param , current_weights):

    PI_0 = sum(T[: , 0]) / (sum(T[: , 0]) + sum(T[:, 1]))
    PI_1 = sum(T[: , 1]) / (sum(T[: , 0]) + sum(T[:, 1]))

    MU_X0 = (sum(T[i , 0] * data[i , 0] for i in range(len(data)))) / sum(T[:, 0])
    MU_X1 = (sum(T[i , 1] * data[i , 0] for i in range(len(data)))) / sum(T[:, 1])

    MU_Z0 = (sum(T[i, 0] * data[i, 1] for i in range(len(data)))) / sum(T[:, 0])
    MU_Z1 = (sum(T[i, 1] * data[i, 1] for i in range(len(data)))) / sum(T[:, 1])

    SIGMA_X0 = sqrt( sum(T[i , 0] * (data[i, 0] - MU_X0) * (data[i ,0] - MU_X0) for i in range(len(data))) / (sum(T[:, 0])))
    SIGMA_X1 = sqrt( sum(T[i , 1] * (data[i, 0] - MU_X1) * (data[i, 0] - MU_X1) for i in range(len(data))) / (sum(T[:, 1])))

    SIGMA_Z0 = sqrt(sum(T[i, 0] * (data[i, 1] - MU_Z0) * (data[i, 1] - MU_Z0) for i in range(len(data))) / (sum(T[:, 0])))
    SIGMA_Z1 = sqrt(sum(T[i, 1] * (data[i, 1] - MU_Z1) * (data[i, 1] - MU_Z1) for i in range(len(data))) / (sum(T[:, 1])))

    RO_0 = (sum( T[i,0] * (((data[i,0] - MU_X0)*(data[i,1] - MU_Z0))/(SIGMA_X0*SIGMA_Z0)) for i in range(len(data)))) / (sum(T[:, 0]))
    RO_1 = (sum( T[i,1] * (((data[i,0] - MU_X1)*(data[i,1] - MU_Z1))/(SIGMA_X1*SIGMA_Z1)) for i in range(len(data)))) / (sum(T[:, 1]))

    return [ [MU_X0,MU_Z0,SIGMA_X0,SIGMA_Z0,RO_0] , [MU_X1,MU_Z1,SIGMA_X1,SIGMA_Z1,RO_1] ] , [PI_0 , PI_1]


################################################################################################

def EM(data , params , weights , nb_iteration):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dessine_normales(data , params , weights , find_bounds(data,params) , ax)
    plt.show()
    for i in range(nb_iteration):
        T = Q_i(data,params,weights)
        params , weights = M_step(data,T,params,weights)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        dessine_normales(data, params, weights, find_bounds(data, params), ax)
        plt.show()

################################################################################################

def EM_Animation(data , params , weights , nb_iteration):
    res = []
    res.append([params , weights])
    for i in range(nb_iteration):
        T = Q_i(data,params,weights)
        params , weights = M_step(data,T,params,weights)
        res.append([params , weights])

    return res


################################################################################################
    
data = read_file ( "tme4_faithful.txt" )

print(normale_bidim ( 1, 2, (1.0,2.0,3.0,4.0,0) ))
print(normale_bidim ( 1, 0, (1.0,2.0,1.0,2.0,0.7) ))

dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.7) )
dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.2) )

# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()

################################################################################################

#current_params = np.array ( [(mu_x, mu_z, sigma_x, sigma_z, rho),   # params 1ère loi normale
#                             (mu_x, mu_z, sigma_x, sigma_z, rho)] ) # params 2ème loi normale
current_params = np.array([[ 3.28778309, 69.89705882, 1.13927121, 13.56996002, 0. ],
                           [ 3.68778309, 71.89705882, 1.13927121, 13.56996002, 0. ]])

# current_weights = np.array ( [ pi_0, pi_1 ] )
current_weights = np.array ( [ 0.5, 0.5 ] )

T = Q_i ( data, current_params, current_weights )
print(T)


current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
T = Q_i ( data, current_params, current_weights )
print("\n\n\n\n",T)


################################################################################################

current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
current_weights = array([ 0.45165145,  0.54834855])
Q = Q_i ( data, current_params, current_weights )

print("\n\n\n" , M_step ( data, Q, current_params, current_weights ))

#
################################################################################################

mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

#EM(data , params , weights , 5)

res_EM = EM_Animation(data , params , weights , 20)

# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

bounds = find_video_bounds ( data, res_EM )


import matplotlib.animation as animation

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print("step animate = %d" % ( i ))

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate,
                               frames = len ( res_EM ), interval=500 )
plt.show ()

# éventuellement, sauver l'animation dans une vidéo
# anim.save('old_faithful.avi', bitrate=4000)


