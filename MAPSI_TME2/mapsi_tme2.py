import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

def bernoulli (p):
    assert (0 <= p <= 1)
    random = np.random.rand() * 100
    assert(0 <= random <= 100)
    if(random < p*100):
        return 1
    return 0

#################################################################

def binomiale (n,p):
    assert(n >= 0)
    assert(0 <= p <= 1)
    res = 0
    for i in range(0,n):
        res += bernoulli(p)
    return res

#################################################################

# planche de Galton

height = 13

array = []
for i in range(0,100000):
    array.append(binomiale(height,0.5))

plt.figure()
plt.hist(array , height)
plt.show()


#################################################################


def normale(k,sigma):
    if(k%2 == 0):
        raise ValueError("Le nombre k doit etre impair")

    Xi = np.linspace(-2*sigma,2*sigma,k)
    Yi = np.zeros(k)
    taille_intervalle = (4*sigma)/k
    nb_tirage = 1000000
    
    for i in range(0,nb_tirage):
        n = np.random.normal(0,sigma*sigma)
        numero_intervalle = 0
        if(-2*sigma <= n <= 2*sigma):
            while(numero_intervalle < k-1 and n > -2*sigma + numero_intervalle*taille_intervalle):
                numero_intervalle += 1
                
            Yi[numero_intervalle] += 1
       
    print("Fin calcul !")
    Yi /= nb_tirage
    plt.figure()
    plt.plot(Xi,Yi)
    plt.show()

    return Yi

#Yi = normale(51,1)


#################################################################

def proba_affine(k,slope):
    if(k%2 == 0):
        raise ValueError("Le nombre k doit etre impair")
    if abs ( slope  ) > 2. / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + str ( 2. / ( k * k ) ) )
        
    Xi = np.arange(0,k)
    Yi = []
    if(slope == 0):
        for i in range(0,k):
            Yi.append(1/k)
    
    else:
        for i in range(0,k):
            Yi.append( (1/k) + ((i-(k-1)/2) * slope))
            
    plt.figure()
    plt.plot(Xi,Yi)
    plt.show()
    
    return Yi

    
#proba_affine(21,0.0042)

#################################################################

def Pxy (Pa, Pb):
    lenA = len(Pa)
    lenB = len(Pb)
    if(lenA != lenB):
        raise ValueError('Les tableaux ne sont pas de la meme taille')
        
    res = np.zeros((lenA,lenB))
    
    for i in range(0,lenA):
        for j in range(0,lenB):
            res[i][j] = Pa[i]*Pb[j]
            
    return res

PA = np.array ( [0.2, 0.7, 0.1] )
PB = np.array ( [0.4, 0.4, 0.2] )

print("Loi jointe PA,PB:\n", Pxy(PA,PB))


#################################################################

def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()
    
#dessine(Pxy(normale(21,1) , proba_affine(21,0.0042)))

#################################################################

# creation de P(X,Y,Z,T)
    
P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])
    
P_YZ = np.zeros((2,2))

for i in range(0,2):
    for j in range(0,2):
        P_YZ[i][j] = P_XYZT[0][i][j][0] + P_XYZT[0][i][j][1] + P_XYZT[1][i][j][0] + P_XYZT[1][i][j][1]
        
print("P_YZ : \n",P_YZ) 

#################################################################

P_XTcondYZ = np.zeros((2,2,2,2))

for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            for l in range(0,2):
                P_XTcondYZ[i][j][k][l] = P_XYZT[i][j][k][l] / P_YZ[j][k]
                
print("P_XTcondYZ:\n",P_XTcondYZ)

P_XcondYZ = np.zeros((2,2,2))

for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            P_XcondYZ[i][j][k] = P_XTcondYZ[i][j][k][0] + P_XTcondYZ[i][j][k][1]
            
print("P_XcondYZ:\n",P_XcondYZ)     


P_TcondYZ = np.zeros((2,2,2))

for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            P_TcondYZ[i][j][k] = P_XTcondYZ[0][i][j][k] + P_XTcondYZ[1][i][j][k]
            
print("P_TcondYZ:\n",P_TcondYZ)       


if(np.array_equal(P_XTcondYZ , np.dot(P_XcondYZ , P_TcondYZ))):
    print("\nX et T sont independantes conditionnelement à Y et Z\n")
else:
    print("\nX et T sont PAS independantes conditionnelement à Y et Z\n")      


#################################################################
    
P_XYZ = np.zeros((2,2,2))

for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            P_XYZ[i][j][k] = P_XYZT[i][j][k][0] + P_XYZT[i][j][k][1]
            
print("P_XYZ:\n",P_XYZ)

P_X = np.zeros((2))

for i in range(0,2):
    P_X[i] = P_XYZ[i][0][0] + P_XYZ[i][0][1] + P_XYZ[i][1][0] + P_XYZ[i][1][1]
    
print("P_X:\n",P_X)

P_YZ = np.zeros((2,2))

for i in range(0,2):
    for j in range(0,2):
        P_YZ[i][j] = P_XYZ[0][i][j] + P_XYZ[1][i][j] 
        
print("P_YZ:\n",P_YZ)

if(np.array_equal(P_XYZ , np.dot(P_X , P_YZ))):
    print("X et YZ sont indépendantes")
else:
    print("X et YZ ne sont PAS indépendantes")


    