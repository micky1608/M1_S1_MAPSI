import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

plt.close('all')

fname = "dataVelib.pkl"
f= open(fname, 'rb')
data = pkl.load(f)
f.close()

correctStations = np.zeros((1, 6))

# Data cleanup

for station in data:
	arr = station['number'] // 1000
	if 1 <= arr <= 20:
		lat = station['position']['lat']
		lon = station['position']['lng']
		alt = station['alt']
		total = station['bike_stands']
		avail = station['available_bikes']
		correctStations = np.vstack((correctStations, np.array([lat, lon, alt, arr, total, avail])))
correctStations = correctStations[1:, :]

# Probabilities

# P[Ar]
PAr = np.zeros((1, 20))
for station in correctStations:
	PAr[0][int(station[3])-1] += 1
PAr /= len(correctStations)

nIntervals = 100
res = plt.hist(correctStations[:, 2], nIntervals)
alt = res[1]
# P[Al]
PAlt = res[0]/np.sum(res[0])

# P[Sp|Al]
PSpAl = np.zeros(100)
# P[Vd|Al]
PVdAl = np.zeros(100)
for i in np.arange(0, 100, 1):
	lower = alt[i]
	upper = alt[i+1]
	nbSt = 0
	for station in correctStations:
		if lower <= station[2] < upper:
			nbSt += 1
			PSpAl[i] += (station[5] == 0)
			PVdAl[i] += (station[5] >= 2)
	if nbSt != 0:
		PSpAl[i] /= nbSt
		PVdAl[i] /= nbSt

# P[Vd|Ar]
PVdAr = np.zeros(20)
for i in np.arange(1, 20):
	nbSt=0
	for station in correctStations:
		if station[3] == i:
			nbSt += 1
			PVdAr += (station[4] >= 2)
	if nbSt != 0:
		PVdAr /= nbSt

# Histogram
plt.close('all')
plt.figure()
plt.title = "Repartition de l'altitude"
altAxis = (alt[1:]+alt[:-1])/2
values = PAlt/(alt[1]-alt[0])
plt.bar(altAxis, values, alt[1]-alt[0])

plt.show()

plt.figure()
plt.title = "Disponibilite selon altitude"
plt.bar(altAxis, PVdAl, alt[1]-alt[0])
plt.show()


x1 = correctStations[:,1] # recuperation des coordonnées
x2 = correctStations[:,0]
# définition de tous les styles (pour distinguer les arrondissements)
style = [(s,c) for s in "o^+*" for c in "byrmck" ]

# tracé de la figure
plt.figure()
for i in range(1,21):
	ind, = np.where(correctStations[:,3]==i)
	# scatter c'est plus joli pour ce type d'affichage
	plt.scatter(x1[ind],x2[ind],marker=style[i-1][0],c=style[i-1][1],linewidths=0)

plt.axis('equal') # astuce pour que les axes aient les mêmes espacements
plt.legend(range(1,21), fontsize=10)
plt.savefig("carteArrondissement.pdf")
axis = plt.axis()

plt.figure()
ind, = np.where(correctStations[:, 5] == correctStations[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)
ind2, = np.where(correctStations[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)
ind3 = np.delete(np.arange(0,len(x1)), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)
plt.axis(axis)
plt.show()
###########################################################################"

plt.figure()
indalt = np.where(correctStations[:, 2] < np.mean(alt))
stations = correctStations[indalt]
x1 = stations[:,1] # recuperation des coordonnées
x2 = stations[:,0]
ind, = np.where(stations[:, 5] == stations[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)
ind2, = np.where(stations[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)
ind3 = np.delete(np.arange(0,len(x1)), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)
plt.axis(axis)
plt.show()

##########################################################################

plt.figure()
indalt = np.where(correctStations[:, 2] > np.median(alt))
stations = correctStations[indalt]
x1 = stations[:,1] # recuperation des coordonnées
x2 = stations[:,0]
ind, = np.where(stations[:, 5] == stations[:, 4])
plt.scatter(x1[ind], x2[ind],marker='+',c='r', linewidths=0)
ind2, = np.where(stations[:, 5] == 0)
plt.scatter(x1[ind2], x2[ind2],marker='+',c='y', linewidths=0)
ind3 = np.delete(np.arange(0,len(x1)), np.concatenate((ind, ind2)))
plt.scatter(x1[ind3], x2[ind3],marker='+',c='g', linewidths=0)
plt.axis(axis)
plt.show()

alts = correctStations[:, 2]
arrs = correctStations[:, 3]
vds = correctStations[:, 5]
meanAlt = np.mean(alts)
meanArrs = np.mean(arrs)
meanVds = np.mean(vds)
stdAlt = np.std(alts)
stdArrs = np.std(arrs)
stdVds = np.std(vds)

altXvds = alts*vds
arrsXvds = arrs*vds

meanAltVds = np.mean(altXvds)
meanArrVds = np.mean(arrsXvds)

print("Cor(Alt, Vd) : " + str((meanAltVds-meanAlt*meanVds)/(np.std(alts)*np.std(vds))))
print("Cor(Arr, Vd) : " + str((meanArrVds-meanArrs*meanVds)/(np.std(arrs)*np.std(vds))))
