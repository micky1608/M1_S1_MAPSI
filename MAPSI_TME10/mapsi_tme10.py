import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def createData(a,b,N,sigma):
    X = []
    Y = []

    for i in range(N):
        x = np.random.rand()
        y = a*x + b + (np.random.randn() * sigma)
        X.append(x)
        Y.append(y)

    return np.array(X),np.array(Y)

################################################################################

def createDataQuad(a,b,c,N,sigma):
    X = []
    Y = []

    for i in range(N):
        x = np.random.rand()
        y = a * x**2 + b * x + c + (np.random.randn() * sigma)
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)
################################################################################

def showData(X,Y):
    print("Shape X : ",np.shape(X))
    print("Shape Y : ",np.shape(Y))
    plt.scatter(X,Y,10)
    plt.title("Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

################################################################################

def showModelisation(X,Y,alpha,beta,color_point,color_line):
    plt.scatter(X, Y, 10 , color_point)
    plt.title("Data")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(np.linspace(0,1,2),[affine(x,alpha,beta) for x in np.linspace(0,1,2)] , color_line)
    plt.show()

################################################################################

def showModelisationQuad(X,Y,alpha,beta,gamma,color_point,color_line):
    plt.scatter(X, Y, 10 , color_point)
    plt.title("Data")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(np.linspace(0,1,10), [quad(x,alpha,beta,gamma) for x in np.linspace(0,1,10)] , color_line)
    plt.show()
    
################################################################################

def affine(x,a,b):
    return a*x+b

################################################################################

def quad(x,a,b,c):
    return a*x**2 + b*x + c

################################################################################

def estimateParam(X,Y):
    alpha = covariance(X,Y)/variance(X)
    beta = np.mean(Y) - np.mean(X)*covariance(X,Y)/variance(X)
    return alpha,beta

################################################################################

def variance(X):
    return np.mean([(x - np.mean(X))**2 for x in X])

################################################################################

def covariance(X,Y):
    tab = []
    for i in range(len(X)):
        tab.append((X[i] - np.mean(X))*(Y[i] - np.mean(Y)))

    return np.mean(tab)

################################################################################

def gradientResolution(X,Y):
    eps = 5e-4
    nIterations = 1000
    w = np.zeros(X.shape[1]) # init à 0
    #w = [5 , 0] # init proche de la bonne valeur
    #w = [100 , 80] # init loin de la bonne valeur
    allw = [w]
    for i in range(1,nIterations):
        w = allw[i-1] - eps*C_gradient(X[:,0],Y,allw[i-1])
        allw.append(w)

    allw = np.array(allw)
    return allw

################################################################################

# W = a,b
def C(X,Y,W):
    a,b = W
    return sum((Y[i] - a*X[i] - b)**2 for i in range(len(X)))

################################################################################

def C_gradient(X,Y,W):
    a,b = W
    gradient = []

    d = []
    for i in range(len(X)) :
        d.append(-2*X[i]*(Y[i] - b - a*X[i]))
    gradient.append(sum(d))

    d = []
    for i in range(len(X)):
        d.append(-2*(Y[i] - b - a * X[i]))
    gradient.append(sum(d))

    return np.array(gradient)

################################################################################


a = 6.
b = -1.
c = 1
N = 100
sigma = .4

X,Y = createData(a,b,N,sigma)
alpha,beta = estimateParam(X,Y)
showModelisation(X,Y,alpha,beta,"red","blue")

X_matrice = np.hstack((X.reshape(N,1),np.ones((N,1))))

A = np.dot(X_matrice.transpose(),X_matrice)
B = np.dot(X_matrice.transpose(),Y)

w = np.linalg.solve(A,B)

showModelisation(X,Y,w[0],w[1],"red","green")

wstar = np.linalg.solve(X_matrice.T.dot(X_matrice), X_matrice.T.dot(Y)) # pour se rappeler du w optimal

allw = gradientResolution(X_matrice,Y)

################################################################################

# tracer de l'espace des couts
ngrid = 20
w1range = np.linspace(-0.5, 10, ngrid)
w2range = np.linspace(-3, 1.5, ngrid)
w1,w2 = np.meshgrid(w1range,w2range)

cost = np.array([[np.log(((X_matrice.dot(np.array([w1i,w2j]))-Y)**2).sum()) for w1i in w1range] for w2j in w2range])

plt.figure()
plt.contour(w1, w2, cost)
plt.scatter(wstar[0], wstar[1],c='r')
plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
plt.show()

costPath = np.array([np.log(((X_matrice.dot(wtmp)-Y)**2).sum()) for wtmp in allw])
costOpt  = np.log(((X_matrice.dot(wstar)-Y)**2).sum())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, w2, cost, rstride = 1, cstride=1 )
ax.scatter(wstar[0], wstar[1],costOpt, c='r')
ax.plot(allw[:,0],allw[:,1],costPath, 'b+-' ,lw=2 )
plt.show()

################################################################################

Xquad,Yquad = createDataQuad(a,b,c,N,sigma)
Xquad2 = Xquad**2

Xe = np.hstack((Xquad2.reshape(N,1),Xquad.reshape(N,1),np.ones((N,1))))

A = np.dot(Xe.transpose(),Xe)
B = np.dot(Xe.transpose(),Yquad)
w = np.linalg.solve(A,B)

showModelisationQuad(Xquad , Yquad , w[0] , w[1] , w[2] , "red" , "blue")

################################################################################

data = np.loadtxt("winequality-red.csv", delimiter=";", skiprows=1)
N,d = data.shape # extraction des dimensions
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]

X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)

# Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)

################################################################################

print("YT : ",YT)
print("\n*********************************************************************\n")


A = np.dot(X.transpose(),X)
B = np.dot(X.transpose(),Y)
w = np.linalg.solve(A,B)

wine_prediction_linear = np.round(np.dot(XT,w)) # Yi = w[0] * X[i,0] + ... + w[N] * X[i,N]

precision = 0
for i in range(len(YT)):
    if YT[i] == wine_prediction_linear[i]:
        precision += 1

precision /= len(YT)

print("Wine prediction_linear : ",wine_prediction_linear)
print("Precision : ",precision)

################################################################################

print("\n*********************************************************************\n")

X2 = X**2
A = np.dot(X2.transpose(),X2)
B = np.dot(X2.transpose(),Y)
w = np.linalg.solve(A,B)

wine_prediction_quad = np.round(np.dot(XT,w)) # Yi = w[0] * X[i,0]**2 + ... + w[N] * X[i,N]**2

precision = 0
for i in range(len(YT)):
    if YT[i] == wine_prediction_quad[i]:
        precision += 1
    
precision /= len(YT)

print("Wine prediction_quad : ",wine_prediction_quad)
print("Precision : ",precision)





