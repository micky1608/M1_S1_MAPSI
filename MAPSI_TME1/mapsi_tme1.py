import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# ********************* File Reading *********************

fname = "dataVelib.pkl"
f = open(fname, 'rb')
data = pkl.load(f,encoding='latin1')
f.close()

# ********************* Filter data *********************

stations = []

for station in data:
    arr = np.floor(station['number'] / 1000)
    if (arr >=0 and arr <=20):
        stations.append(station)

# ********************* Shape data *********************

nb_stations = len(stations)
mat = np.zeros((nb_stations, 6))

for station in stations:
    indice = stations.index(station)

    mat[indice,0] = station['alt']
    mat[indice,1] = station['position']['lat']
    mat[indice,2] = station['position']['lng']
    mat[indice,3] = np.floor(station['number']/1000)
    mat[indice,4] = station['bike_stands']
    mat[indice,5] = station['available_bikes']

# ********************* P[Arr] *********************

Parr = np.zeros((20))

for station in mat:
    arr = int(station[3])
    Parr[arr-1] = Parr[arr-1] + 1

Nb_station_arrondissement = np.array(Parr)
Parr /= nb_stations
Parr = np.vstack((np.arange(1,21,1) , Parr))

assert (np.sum(Parr[1,:]) == 1)

# ********************* P[Al] *********************

nItervalles = 30
min_altitude = min(mat[:,0])
max_altitude = max(mat[:,0])
taille_intervalles = ( max_altitude - min_altitude) / nItervalles

res = plt.hist(mat[:,0], nItervalles)
# res[0] effectif dans les intervalles
# res[1] definition des intervalles (ATTENTION: 31 valeurs)

Pal = res[0]/nb_stations

assert (np.sum(Pal) == 1)


# ********************* P[Sp|Al] *********************


P_SpAl = np.zeros((30))
Nb_station_altitude = np.zeros((30))

for station in mat:
    Sp = 0
    if(station[4] == station[5]):
        Sp = 1


    numero_intervalle = 1;
    while(station[0] > min_altitude + (numero_intervalle*taille_intervalles)):
        numero_intervalle += 1

    P_SpAl[numero_intervalle - 1] += Sp
    Nb_station_altitude[numero_intervalle - 1] += 1

for i in range(0 , 30 , 1):
    if (Nb_station_altitude[i] != 0):
        P_SpAl[i] /= Nb_station_altitude[i]


assert (sum(Nb_station_altitude) == nb_stations)


# ********************* P[Vd|Al] *********************

P_VdAl = np.zeros((30))

for station in mat:
    Vd = 0
    if(station[5] >= 2):
        Vd = 1

    numero_intervalle = 1;
    while (station[0] > min_altitude + (numero_intervalle * taille_intervalles)):
        numero_intervalle += 1

    P_VdAl[numero_intervalle - 1] += Vd

for i in range(0 , 30 , 1):
    if (Nb_station_altitude[i] != 0):
        P_VdAl[i] /= Nb_station_altitude[i]


# ********************* P[Vd|Ar] *********************

P_VdAr = np.zeros((20))

assert (sum(Nb_station_arrondissement) == nb_stations)

for station in mat:
    Vd = 0
    if(station[5] >= 2):
        Vd=1

    P_VdAr[int(station[3]) - 1] += Vd

for i in range(0 , 20 , 1):
    if (Nb_station_arrondissement[i] != 0):
        P_VdAr[i] /= Nb_station_arrondissement[i]



# ********************* Histogram P[Al] *********************

alt = res[1]

plt.close('all')
plt.figure()
plt.title = "Repartition de l'altitude"
plt.bar((alt[1:]+alt[:-1])/2 , Pal/taille_intervalles , taille_intervalles)
plt.show()


# ********************* Stations repartition *********************



x1 = mat[:,2] # get the coordinates
x2 = mat[:,1]
# define the style
style = [(s,c) for s in "o^+*" for c in "byrmck" ]

# trace the figure
plt.figure()
for i in range(1,21):
    ind, = np.where(mat[:,3]==i)
    plt.scatter(x1[ind],x2[ind],marker=style[i-1][0],c=style[i-1][1],linewidths=0)

plt.axis('equal') # for the axis to have the same spacing
plt.legend(range(1,21), fontsize=10)
plt.savefig("carteArrondissements.pdf")
plt.show()

############################################################################

plt.figure()
ind, = np.where(mat[:, 5] == mat[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)

ind2, = np.where(mat[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)

ind3 = np.delete(np.arange(0,nb_stations), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)
plt.axis('equal')
plt.show()


############################################################################

plt.figure()
alt_mean = np.mean(mat[:,0])
ind4 = np.where(mat[:,0] < alt_mean)
stations = mat[ind4]

x1 = stations[:,2]
x2 = stations[:,1]

nb_stations_alt_inf_moyenne = len(x1)

ind, = np.where(stations[:, 5] == stations[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)

ind2, = np.where(stations[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)

ind3 = np.delete(np.arange(0,nb_stations_alt_inf_moyenne), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)

plt.show()

############################################################################

alt_mediane = np.median(mat[:,0])
ind5 = np.where(mat[:,0] > alt_mediane)
stations = mat[ind5]

x1 = stations[:,2]
x2 = stations[:,1]

nb_stations_alt_sup_mediane = len(x1)

ind, = np.where(stations[:, 5] == stations[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)

ind2, = np.where(stations[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)

ind3 = np.delete(np.arange(0,nb_stations_alt_sup_mediane), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)

plt.show()

############################################################################

altitudes = mat[:,0]
arrondissements = mat[:,3]
vds = mat[:,5]

E_Alt = np.mean(altitudes)
E_Arrs = np.mean(arrondissements)
E_Vds = np.mean(vds)


sigma_Alt = np.std(altitudes)
sigma_Arrs = np.std(arrondissements)
sigma_Vds = np.std(vds)

alt_vds = altitudes*vds
arr_vds = arrondissements*vds

E_alt_vds = np.mean(alt_vds)
E_arr_vds = np.mean(arr_vds)

correlation_alt_vds = (E_alt_vds - E_Alt*E_Vds)/(sigma_Alt*sigma_Vds)
correlation_arr_vds = (E_arr_vds - E_Arrs*E_Vds)/(sigma_Arrs*sigma_Vds)




