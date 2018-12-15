import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    plt.show()
    return

############################################################################################################

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

############################################################################################################

def discretise(X,d):
    X_discret = []
    intervalle = 360 / d
    for sig in X:
        X_discret.append(np.floor(sig/intervalle))

    return X_discret

############################################################################################################

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

############################################################################################################

def learnMarkovModel(Xc,d):
    A = np.zeros((d,d))
    Pi = np.zeros(d)

    for X in Xc:
        for i in range(len(X) - 1):
            A[int(X[i]),int(X[i+1])] += 1
            if i==0:
                Pi[int(X[i])] += 1

    A = A / np.maximum(A.sum(1).reshape(d, 1), 1)  # normalisation
    Pi = Pi/Pi.sum()


    return Pi,A

############################################################################################################

def learnMarkovModel_ones(Xc,d):
    A = np.ones((d,d))
    Pi = np.ones(d)

    for X in Xc:
        for i in range(len(X) - 1):
            A[int(X[i]),int(X[i+1])] += 1
            if i==0:
                Pi[int(X[i])] += 1

    A = A / np.maximum(A.sum(1).reshape(d, 1), 1)  # normalisation
    Pi = Pi/Pi.sum()


    return Pi,A

############################################################################################################

def probaSequence(s,Pi,A):

    res = Pi[int(s[0])]

    for k in range(0 ,len(s) - 1):
        proba_transition = A[int(s[k]) , int(s[k+1])]
        res *= proba_transition

    if res != 0:
        return np.log(res)
    else:
        return np.inf*-1

############################################################################################################

def random_state(distribution):
    distribution = np.array(distribution)
    t = np.random.rand()
    sc = distribution.cumsum()
    res = 0
    while(t > sc[res]):
        res +=1

    return res


############################################################################################################

def generate(Pi,A,N):
    states = []
    states.append(random_state(Pi))

    for i in range(N):
        s = random_state(A[states[len(states) - 1]])
        states.append(s)

    return states

############################################################################################################

# old version = python 2
# data = pkl.load(file("ressources/lettres.pkl","rb"))
# new :
with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées

index = groupByLabel(Y)

d3 = 3
Xd3 = discretise(X, d3)
Xd3 = np.array(Xd3)

d20 = 20
Xd20 = discretise(X, d20)
Xd20 = np.array(Xd20)

d50 = 50
Xd50 = discretise(X, d50)
Xd50 = np.array(Xd50)

models3 = []
models20 = []
models50 = []

for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models3.append(learnMarkovModel(Xd3[index[cl]], d3))
    models20.append(learnMarkovModel(Xd20[index[cl]],d20))
    models50.append(learnMarkovModel(Xd50[index[cl]], d50))


proba3 =  np.array([[probaSequence(Xd3[i], models3[cl][0], models3[cl][1]) for i in range(len(Xd3))] for cl in range(len(np.unique(Y)))])
proba20 = np.array([[probaSequence(Xd20[i], models20[cl][0], models20[cl][1]) for i in range(len(Xd20))] for cl in range(len(np.unique(Y)))])
proba50 = np.array([[probaSequence(Xd50[i], models50[cl][0], models50[cl][1]) for i in range(len(Xd50))] for cl in range(len(np.unique(Y)))])


Ynum = np.zeros((Y.shape))
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num


pred3 = proba3.argmax(0)
pred20 = proba20.argmax(0)
pred50 = proba50.argmax(0)

print("Percentage of correct classification with d = 3 : ", np.where(pred3 != Ynum, 0., 1).mean())
print("Percentage of correct classification with d = 20 : ", np.where(pred20 != Ynum, 0., 1).mean())
print("Percentage of correct classification with d = 50 : ", np.where(pred50 != Ynum, 0., 1).mean())

print("\n************ Separation data learning and data test ************")



itrain,itest = separeTrainTest(Y,0.8)

index_learning = []
index_test = []

for i in itrain:
    index_learning += i.tolist()
for i in itest:
    index_test += i.tolist()

print("\n\t***** Counting from 0 *****\n")

models_train3 = []
models_train20 = []
models_train50 = []

for cl in range(len(itrain)):
    models_train3.append(learnMarkovModel(Xd3[itrain[cl]], d3))
    models_train20.append(learnMarkovModel(Xd20[itrain[cl]], d20))
    models_train50.append(learnMarkovModel(Xd50[itrain[cl]], d50))


proba_train3 = np.array([[probaSequence(Xd3[i], models_train3[cl][0], models_train3[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])
proba_train20 = np.array([[probaSequence(Xd20[i], models_train20[cl][0], models_train20[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])
proba_train50 = np.array([[probaSequence(Xd50[i], models_train50[cl][0], models_train50[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])

pred_train3 = proba_train3.argmax(0)
pred_train20 = proba_train20.argmax(0)
pred_train50 = proba_train50.argmax(0)

Ynum_train = Ynum[index_test]

print("Percentage of correct classification with d = 3 : ", np.where(pred_train3 != Ynum_train, 0., 1).mean())
print("Percentage of correct classification with d = 20 : ", np.where(pred_train20 != Ynum_train, 0., 1).mean())
print("Percentage of correct classification with d = 50 : ", np.where(pred_train50 != Ynum_train, 0., 1).mean())


print("\n\t***** Counting from 1 *****\n")

models_train3 = []
models_train20 = []
models_train50 = []

for cl in range(len(itrain)):
    models_train3.append(learnMarkovModel_ones(Xd3[itrain[cl]], d3))
    models_train20.append(learnMarkovModel_ones(Xd20[itrain[cl]], d20))
    models_train50.append(learnMarkovModel_ones(Xd50[itrain[cl]], d50))


proba_train3 = np.array([[probaSequence(Xd3[i], models_train3[cl][0], models_train3[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])
proba_train20 = np.array([[probaSequence(Xd20[i], models_train20[cl][0], models_train20[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])
proba_train50 = np.array([[probaSequence(Xd50[i], models_train50[cl][0], models_train50[cl][1]) for i in index_test] for cl in range(len(np.unique(Y)))])

pred_train3 = proba_train3.argmax(0)
pred_train20 = proba_train20.argmax(0)
pred_train50 = proba_train50.argmax(0)

Ynum_train = Ynum[index_test]

print("Percentage of correct classification with d = 3 : ", np.where(pred_train3 != Ynum_train, 0., 1).mean())
print("Percentage of correct classification with d = 20 : ", np.where(pred_train20 != Ynum_train, 0., 1).mean())
print("Percentage of correct classification with d = 50 : ", np.where(pred_train50 != Ynum_train, 0., 1).mean())

############################################################################################################

Nc = 26

conf_train3 = np.zeros((Nc,Nc))
conf_train20 = np.zeros((Nc,Nc))
conf_train50 = np.zeros((Nc,Nc))

for i in range(len(Ynum_train)):
    conf_train3[int(pred_train3[i]) , int(Ynum_train[i])] += 1
    conf_train20[int(pred_train20[i]), int(Ynum_train[i])] += 1
    conf_train50[int(pred_train50[i]), int(Ynum_train[i])] += 1

CONF = np.array([conf_train3,conf_train20,conf_train50])

for conf in CONF:
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Y))
    plt.yticks(np.arange(26),np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.show()


############################################################################################################




models = []

for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd20[index[cl]], d20))

newa = generate(models[0][0],models[0][1], 25) # generation d'une séquence d'états
intervalle = 360./d20 # pour passer des états => valeur d'angles
newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
tracerLettre(newa_continu)
