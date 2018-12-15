import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random


# get a random point in [-m,m]
def tirage(m):
    size_intervalle = 2*m
    x = np.random.rand()
    y = np.random.rand()
    x = size_intervalle*x - m
    y = size_intervalle*y - m
    return [x,y]

####################################################################################

def monteCarlo(N):
    X = []
    Y = []

    nb_point_cercle = 0

    for i in range(N):
        x,y = tirage(1)

        if x**2 + y**2 <= 1:
            nb_point_cercle +=1

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    return (4*nb_point_cercle)/N,X,Y

####################################################################################

plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(1e4))

print("Pi estimation : ",pi)

# trace les points dans le cercle et hors du cercle

x2 = np.square(x)
y2 = np.square(y)

dist = x2 + y2
plt.plot(x[dist <=1], y[dist <=1], "go")
plt.plot(x[dist>1], y[dist>1], "ro")
plt.show()


####################################################################################

def swapF(dico):

    newDico = {}

    c1 = random.choice(list(dico.keys()))
    c2 = random.choice(list(dico.keys()))

    while c2 == c1:
        c2 = random.choice(list(dico.keys()))

    for key in dico:
        if key == c1:
            newDico[key] = dico[c2]
        elif key == c2:
            newDico[key] = dico[c1]
        else:
            newDico[key] = dico[key]

    return newDico

####################################################################################

def decrypt(message , dico):
    decryptedMessage = ""

    for letter in message:
        if dico.keys().__contains__(letter):
            decryptedMessage += dico[letter]

    return decryptedMessage

####################################################################################

def logLikelihood(message , mu , A , chars2index):

    logL = np.log(mu[chars2index[message[0]]])

    for i in range(1,len(message)):
        logL += np.log(A[chars2index[message[i-1]] , chars2index[message[i]]])

    return logL

####################################################################################

def identityDecoding(count):
    tau = {}
    for k in list(count.keys ()):
        tau[k] = k
    return tau
####################################################################################

def freqDecoding(count):

    freqKeys = np.array(list(count.keys()))
    freqVal  = np.array(list(count.values()))
    rankFreq = (-freqVal).argsort()

    cles = np.array(list(set(secret2))) # tous les caracteres de secret2
    rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))

    tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])

    return tau_init

####################################################################################

def MetropolisHastings(encryptedMessage , mu , A , decodingFunction , N , chars2index):

    decodedMessage = decrypt(encryptedMessage , decodingFunction)
    logL = logLikelihood(decodedMessage, mu , A , chars2index)

    print("Decoding ....")
    for i in range(N):
        newDecodingFunction = swapF(decodingFunction)
        newDecodedMessage = decrypt(encryptedMessage , newDecodingFunction)
        newlogL = logLikelihood(newDecodedMessage , mu , A , chars2index)

        if newlogL > logL or (logL - newlogL < 100 and np.random.rand() <= 1/np.power(np.e,logL-newlogL)):
            decodingFunction = newDecodingFunction
            decodedMessage = newDecodedMessage
            logL = newlogL

    return decodedMessage

####################################################################################

def MCMC(N,p):
    
    nb_point_cercle = 0
    epsilon = 0.002
    nbLigne = 2/epsilon
    
    print("epsillon : ",epsilon)
    print("nb ligne = nb col = ",nbLigne)
    x,y = tirage(1)
    

    x_discret = discretise(x,epsilon)
    y_discret = discretise(y,epsilon)
    print(x_discret," , ",y_discret)

    for i in range(N):
        random = tirage(1)
        new_X_discret = discretise(random[0],epsilon)
        new_Y_discret = discretise(random[1],epsilon)
        
        if new_X_discret <= nbLigne and new_Y_discret <=nbLigne:
            print(new_X_discret," , ",new_Y_discret)
            
            centerX = float((x_discret*epsilon + (x_discret + 1)*epsilon) / 2)
            centerY = float((y_discret*epsilon + (y_discret + 1)*epsilon) / 2)
            
            if centerX**2 + centerY**2 <= 1:
                    nb_point_cercle += 1
                    
            x_discret = new_X_discret
            y_discret = new_Y_discret
    
    return (4*nb_point_cercle)/N
        
####################################################################################
    
def discretise(x,epsilon):
    x_discret = 0
    
    if x > 0:
        while x > epsilon*x_discret:
            x_discret += 1
    elif x < 0:
        while x < epsilon*x_discret:
            x_discret -= 1
    
    return x_discret
    
####################################################################################

# si vos fichiers sont dans un repertoire "ressources"
with open("countWar.pkl", 'rb') as f:
    (count, mu, A) = pkl.load(f, encoding='latin1')

with open("secret.txt", 'r') as f:
    secret = f.read()[0:-1] # -1 pour supprimer le saut de ligne

with open("secret2.txt", 'r') as f:
    secret2 = f.read()[0:-1]
    
with open("fichierHash.pkl" , 'rb') as f:
    chars2index = pkl.load(f, encoding='latin1')

tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }

print("initial dict : ",tau)
print("dict after swap : ",swapF(tau))

print("Decrypt 'aabcd' with tau : ", decrypt ( "aabcd", tau))
print("Decrypt 'dcba' with tau : ",decrypt ( "dcba", tau ))

#chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))

print("char2index : ",chars2index)

print("logLikelihood 'abcd' : ",logLikelihood("abcd" , mu , A , chars2index))
print("logLikelihood 'dcba' : ",logLikelihood("dcba" , mu , A , chars2index))

secret2Decoded = MetropolisHastings(secret2, mu, A, freqDecoding(count), 50000, chars2index)

print("Secret2 decoded : ",secret2Decoded)
print("logL : ", logLikelihood(secret2Decoded , mu , A , chars2index))

print("Pi estimation : ",MCMC(int(1e4),1))