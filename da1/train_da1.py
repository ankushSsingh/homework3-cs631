import csv
import numpy as np
from numpy import array 
from scipy.linalg import svd,inv
import math as m
import matplotlib.pyplot as plt


x = np.zeros([250,250])
full = np.zeros([250,4550])

with open("./xmv10_359_data_1.csv", "r") as ifile:
	reader = csv.reader(ifile)
	s = list(reader)
	
# To plot the orignial sensor readings
l = len(s)
xtrain = np.zeros(4800)
I=[0]*4800
for i in range(4800):
	xtrain[i] = s[i][14]
	I[i]=i
plt.plot(I,xtrain,c="blue")
plt.ylabel("Sensor Reading (XMEAS[15])")
plt.xlabel("Observation Points")
plt.title("Observation Points vs Original Readings for 4800 obervations in DA1 ")
plt.show()
	

i=0
for i in range(250):
	x[:,i] = xtrain[i:250+i]

i=0
for i in range(4550):
	full[:,i] = xtrain[i:250+i]

#	Singular Value Decomposition of X 
u,s,vt = svd(x)

#To plot the scree_plot
sc=np.zeros(251)
sc_full = np.dot(s,s)
sc[0]=0
itr=[0]*251
for i in range(1,251):
	sc[i]= sc[i-1] + s[i-1]*s[i-1]/sc_full
	itr[i]=i

plt.plot(itr[:5],sc[:5],c="blue")
plt.xlabel("Iterations")
plt.ylabel("Normalised Cumulative Squared Eigen-Value")
plt.title("DA1")

plt.show() 

# sample_mean calculated
mean = np.mean(x,axis=1)



#choose the statistical dimension r
r=1

# Projection Matrix calculated
U = np.zeros([250,r])
for i in range(r):
	U[:,i] = u[:,i]
b = np.matmul(U.T,U)
p = np.matmul(U,np.matmul(inv(b),U.T))

# Projected Mean calculated
pmean = np.matmul(U.T,mean)
# print(pmean)


# Departure Score = || pmean - U.T * x || ^ 2
loop = 4550
d = [0]*loop
iter = [0]*loop
for i in range(loop):
	px = np.matmul(U.T,full[:,i])
	d[i] = m.pow(pmean - px ,2)
	iter[i]=i


#Threshold_Score i.e. threshold = max ||d|| ^ 2 in validation region
threshold = max(d[501:3750])
print(threshold)


#To plot the departure_scores
plt.plot(iter,d,c ="red")
t=[threshold]*loop
plt.plot(iter,t,linestyle="--",c="black")
plt.xlabel('Time Stamps')
plt.ylabel('Departure Scores')
plt.title('DA1 ')
plt.show()



