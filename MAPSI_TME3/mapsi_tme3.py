import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

#################################################################################################

def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace(0, 9, 10)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride=1, cstride=1)
    fig.tight_layout()
    ax.view_init(elev=60., azim=-45)
    plt.show()

#################################################################################################
def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

#################################################################################################

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()


#################################################################################################

def learnML_class_parameters(tab_class):
    nb_image = len(tab_class)
    MU = np.zeros(256)
    SIGMA = np.zeros(256)

    for p in range(0,256):
        for i in range(0,nb_image):
            MU[p] += tab_class[i][p]

        MU[p] /= nb_image

        for i in range(0, nb_image):
            SIGMA[p] += (tab_class[i][p] - MU[p])*(tab_class[i][p] - MU[p])

        SIGMA[p] /= nb_image

    return MU,SIGMA

#################################################################################################

def learnML_all_parameters(data):
    all_param = []

    for i in range(0,10):
        param = learnML_class_parameters(data[i])
        all_param.append(param)

    return all_param

#################################################################################################


def log_likelihood(image , param):
    P = 0

    for i in range(0,256):
        Xi = image[i]
        mu = param[0][i]
        sigma2 = param[1][i]
        if sigma2 != 0:
            P -= 0.5 * ( math.log(2*math.pi*sigma2) + ((Xi - mu)*(Xi-mu))/sigma2)

    return P

#################################################################################################

def log_likelihoods(image , param):
    return [ log_likelihood (image, param[i] ) for i in range (10) ]

#################################################################################################

def classify_image(image , param):
    return np.argmax(log_likelihoods(image,param))

#################################################################################################

def classify_all_images(data , param):
    classification = np.zeros((10,10))

    for i in range(0,10):
        for j in range(0,len(data[i])):
            estimated_class = classify_image(data[i][j] , param)
            classification[i][estimated_class] += 1
        classification[i] /= len(data[i])

    return classification

#################################################################################################

print("Reading data file ...")
data = read_file("tme3_usps_train.txt")

print("Reading data test file ...")
data_test = read_file("tm3_usps_test.txt")

print("Learning parameters ...")
parameters = learnML_all_parameters (data)

print("Classifying images ...")
T = classify_all_images(data_test , parameters)

dessine(T)






