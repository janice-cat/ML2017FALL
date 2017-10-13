import sys
import pandas as pd
import numpy as np

sys.argv[1]
TrainingData = pd.read_csv(sys.argv[1],encoding = "big5", header = None)
TD_raw = pd.DataFrame(TrainingData)

'''
	TIME
0	AMB_TEMP
1	CH4
2	CO
3	NMHC
4	NO
5	NO2
6	NOx
7	O3
8	PM10
9	PM2.5
10	RAINFALL
11	RH
12	SO2
13	THC
14	WD_HR
15	WIND_DIREC
16	WIND_SPEED
17	WS_HR
'''


np.set_printoptions(threshold=np.inf)

TD = [[[0 for x in range(18)] for y in range(9)] for z in range(240)]
id_value = [0 for z in range(240)]


for row in TD_raw.itertuples():
	Index = row[0]
	C = Index%18
	for i in range(3,12,1):
		R = i-3
		Z = Index//18
		if C == 10 and row[i] == "NR":
		      TD[Z][R][C] = -1
		else:
		      TD[Z][R][C] = float(row[i])
		id_value[Z] = row[1]


##Container and Parameters
Month = []
feature_num = 1 ##### M all:18 , square :19 , PM2.5:1
hour_num = 9  ##### M 4 5 6
N_test = 240

#Add Test Data
for i in range(N_test): ##### M
        '''
        #all
        Month.append(np.matrix(TD[i])) #####VD

        
        #source of contaimination
        temp = np.array(TD[i])
        TD_A1 = temp[:,0:18]
        TD_A2 = np.square([temp[:,9]]).T
        #print(TD_A2)
        #print(len(TD_A2))
        A1A2 = np.hstack((TD_A1,TD_A2))
        Month.append(np.matrix(A1A2))

        '''
        #PM2.5
        temp = np.array(TD[i])
        Month.append(temp[:,9])
        

#function
bias = np.matrix( [[ 1.9285981953764597]] )
weight_matrix = np.matrix([[-0.03694406, -0.01189283,  0.1860495 , -0.22531685, -0.03540377,
          0.49763223, -0.55038813, -0.03310659,  1.13029649]])


#predict function
def predict(k,i):
        #x_matrix = Month[k][i:i+hour_num,:]    #####    Other
        x_matrix = Month[k][i:i+hour_num]     #####    PM2.5
        y_hat = bias+np.sum(np.multiply(weight_matrix,x_matrix))
        return (y_hat)  

print("id,value")

for k in range(N_test):
        #x_matrix = Month[k][j:j+hour_num,:]    #####    Other
        x_matrix = Month[k][0:0+hour_num]     #####    PM2.5
        p = predict(k,0)
        print("%s,%f"%(id_value[k],p))
