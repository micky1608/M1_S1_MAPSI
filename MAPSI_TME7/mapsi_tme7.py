import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def discretise(X,d):
    X_discret = []
    intervalle = 360 / d
    for sig in X:
        X_discret.append(np.floor(sig/intervalle))

    return X_discret

###############################################################################################

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index



###############################################################################################

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

###############################################################################################

def initGD(X,N):
    S = []
    for Xi in X:
        S.append(np.floor(np.linspace(0,N-.00000001,len(Xi))))

    return S

###############################################################################################

# K = all the observations
# S = all the states
# N = Number of states
# K = discretisation

def learnHMM(X,S,N,K,initTo0 = False):

    if initTo0:
        A = np.zeros((N, N))
        B = np.zeros((N, K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N, N)) * eps
        B = np.ones((N, K)) * eps
        Pi = np.ones(N) * eps

    for i in range(len(X)):
        Xi = X[i]
        Si = S[i]

        for k in range(len(Si)):
            if k<len(Si)-1:
                A[int(Si[k]) , int(Si[k+1])] += 1
            if k==0:
                Pi[int(Si[0])] += 1
            B[int(Si[k]), int(Xi[k])] += 1

    A = A / np.maximum(A.sum(1).reshape(N, 1), 1)  # normalisation
    B = B / np.maximum(B.sum(1).reshape(N, 1), 1)  # normalisation
    Pi = Pi / Pi.sum()

    return Pi,A,B

###############################################################################################

def viterbi(x,Pi,A,B):
    nb_state = len(Pi)
    nb_observation = len(x)

    delta_previous = []
    delta = np.zeros(nb_state)
    psi = []
    psi_i = np.zeros(nb_state)


    for i in range(nb_state):
        delta_previous.append(np.log(Pi[i]) + np.log(B[i,int(x[0])])) if Pi[i] != 0 and B[i,int(x[0])] != 0 else delta_previous.append(np.inf*-1)


    for i in range(1,nb_observation):
        for j in range(nb_state): # for k in range(len(delta))
            v = []
            for k in range(nb_state):
                if A[k, j] != 0:
                    v.append(delta_previous[k] + np.log(A[k,j]))
                else:
                    v.append(np.inf*-1)

            if B[j, int(x[i])] != 0:
                delta[j] = max(v) + np.log(B[j,int(x[i])])
            else:
                delta[j] = np.inf*-1

            psi_i[j] = np.argmax(v)

        psi.append(psi_i[np.argmax(delta)])

        delta_previous = delta

        delta = np.zeros(nb_state)
        psi_i = np.zeros(nb_state)

    psi.append(float(np.argmax(delta_previous)))

    return max(delta_previous),psi


###############################################################################################

def log_prob_v2(x,Pi,A,B):
    nb_state = len(Pi)
    nb_observation = len(x)

    alpha_previous = []
    alpha = np.zeros(nb_state)

    for i in range(nb_state):
        alpha_previous.append(Pi[i] * B[i, int(x[0])])

    for i in range(1, nb_observation):
        for j in range(nb_state):
            alpha[j] = sum(alpha_previous[k] * A[k, j] for k in range(nb_state)) * B[j, int(x[i])]

        alpha_previous = alpha
        alpha = np.zeros(nb_state)

    return np.log(max(alpha_previous))

###############################################################################################


np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))
index = groupByLabel(Y)
nCl = 26

K=10
N=5

Xd = discretise(X,K)
Xd = np.array(Xd)

S = initGD(Xd,N)
S = np.array(S)

Pi,A,B = learnHMM(Xd[Y=='a'],S[Y=='a'],N,K,True)

print("Pi = \n",Pi)
print("A = \n",A)
print("B = \n",B)


delta,psi = viterbi(Xd[0],Pi,A,B)

print("Most probable sequence log-probability : ",delta)
print("Most probable sequence : ",psi)

print("alpha : ",log_prob_v2(Xd[0],Pi,A,B))

