"""
Created on Fri Oct 28 15:11:12 2022
@author: ssvvkk
"""

from wsgiref.handlers import format_date_time
import pandas as pd
import os # for rawdata loading
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# constants

Horizontal = 1
Vertical = 0
Left = 1
Right = 0

shoulder = 1
elbow = 2
hip = 3
knee = 4

isHorizontal = True

#User selection

file_readfolder = ".\sensordata"
file_writefolder = "dataprocessed"
file_clapfolder= "claptimeframe"
file_splitfolder = "splitbyclap"
file_startstring = "\\"
file_sensorfile = "221111 kaist_dancer1_magnet_mov2" #ex: kaist_dancer1_magnet_mov1
ismagnet = True
decimal_points = 4
file_writestring = "_angle"
file_clapstring = "_clap"
file_extension_clap = ".csv"
file_extension = ".txt"

poly_degree = 5
poly_bias = True

#check whether the sensor is attatched on joint
""" class node:
    def __init__(self, nodeNum, Horizontal, Left, position, x_intercept, x_poly):
        self.nodeNum = nodeNum
        self.isHorizontal = Horizontal
        self.isLeft = Left
        self.isRight = ~ Left
        self.position = position
        self.x_intercept = x_intercept
        self.x_poly = x_poly
    def add_poly(self, x_interept, x_poly):
        self.x_intercept = x_interept
        self.x_poly = x_poly
 """
def poly_regression(poly_degree, poly_bias, ismagnet, isHorizontal):
    
    ## angle file reading
    new_index = ['time','x1','y1','x2','y2','deltax','deltay','bx','by','bz','norm(b)']
    
    if ismagnet:
        if isHorizontal:
            df_angle = df_angle_o_h = pd.read_csv("angledata/magO_1_h.txt", header=None, names=new_index)
        else:
            df_angle = df_angle_o_v = pd.read_csv("angledata/magO_1_v.txt", header=None, names=new_index)
    else:
        if isHorizontal:
            df_angle = df_angle_x_h = pd.read_csv("angledata/magX_1_h.txt", header=None, names=new_index)
        else:
           df_angle =  df_angle_x_v = pd.read_csv("angledata/magX_1_v.txt", header=None, names=new_index)


    y_axis = 'deltax'
    x_axis1 = 'bx'
    x_axis2 = 'by'
    x_axis3 = 'bz'

    #sensor and magnetometer mappling
    # for horizontal attatchment    --> use Bz, ('By' not use)
    # for vertical attatchment      --> use Bx, ('By' not use)

    if isHorizontal:
        cal_x_axis1 = x_axis1
    else:
        cal_x_axis1 = x_axis1

    df_angle.reindex(new_index, fill_value=0)


    ## filtering - remove sensor outlier data
    q_low = df_angle[cal_x_axis1].quantile(0.30)
    q_hi  = df_angle[cal_x_axis1].quantile(0.70)

    df = df_angle[(df_angle[cal_x_axis1] < q_hi) & (df_angle[cal_x_axis1] > q_low)]

    # change to numpy array and reduce array size
    
    y =df[y_axis].to_numpy()
    x = df[cal_x_axis1].to_numpy()

    
    poly = PolynomialFeatures(degree=poly_degree, include_bias=poly_bias)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    poly_coef = poly_reg_model.coef_
    poly_intercept =poly_reg_model.intercept_

    x_i = 1.0
    x_poly=[x_i**0, x_i**1, x_i**2, x_i**3, x_i**4, x_i**5]

    #y_estimated = poly_intercept + x_poly*poly_coef 

    plt.figure(figsize=(10, 6))
    plt.title("polynomial regression)", size=16)
    plt.scatter(x, y)
    plt.plot(x, y_predicted, c="red")
    #plt.ylim(0,1000)
    #plt.show()

    return poly_intercept, poly_coef

if ismagnet:
    mag_intercept_H, mag_coef_H = poly_regression(poly_degree, poly_bias, ismagnet=True, isHorizontal=True)
    mag_intercept_V, mag_coef_V = poly_regression(poly_degree, poly_bias, ismagnet=True, isHorizontal=False)
    intercept_H = mag_intercept_H
    coef_H = mag_coef_H
    intercept_V = mag_intercept_V
    coef_V = mag_coef_V
else:
    coil_intercept_H, coil_coef_H = poly_regression(poly_degree, poly_bias, ismagnet=False, isHorizontal=True)
    coil_intercept_V, coil_coef_V = poly_regression(poly_degree, poly_bias, ismagnet=False, isHorizontal=False)
    intercept_H = coil_intercept_H
    coef_H = coil_coef_H
    intercept_V = coil_intercept_V
    coef_V = coil_coef_V

""" node1 = node(1, Horizontal, Left, shoulder, intercept_H, coef_H)
node2 = node(2, Horizontal, Left, elbow, intercept_H, coef_H)
node3 = node(3, Vertical, Left, hip, intercept_V, coef_V)
node4 = node(4, Vertical, Left, knee, intercept_H, coef_H)
node6 = node(5, Horizontal, Right, shoulder, intercept_H, coef_H)
node7 = node(6, Horizontal, Right, elbow, intercept_H, coef_H)
node8 = node(7, Vertical, Right, hip, intercept_V, coef_V)
node9 = node(8, Vertical, Right, knee, intercept_H, coef_H) """

# for dirpath, dirnames, files in os.walk('.\sensordata', topdown=True): 
#     # os.walk method() 
#     # in current root: os.walk('.') 
#     # specific root: os.walk(path)
#     print(f'Found directory: {dirpath}')
#     for file in files:
#         print(dirpath)      # current path name
#         #print(dirnames)     # directories in current path
#         #print(files)        # files in current path
        
#         #if file_idx
#         print(file)
#         filename, ext = os.path.splitext(file)
#         print(filename)
#         print(ext)
#         #cur_fullpath = os.path.join(dirpath, "\", file_idx)


for file in os.listdir(file_readfolder):
        #if file_idx
        #print(file)
        filename, ext = os.path.splitext(file)
        file_sensorfile = filename
        #print(filename)
        #print(ext)

        if ext: #if iteration item is not a folder

            if "magnet" in filename:
                ismagnet = True
            else: 
                ismagnet = False

            #read sensor data
            #sensor_index = ['Time,Node,Bx,By,Bz,|B|mag']
            sensor_index = ['time','node','Bx','By','Bz','|B|mag','blank']
            df_sensordata = pd.read_csv(file_readfolder + file_startstring +  file_sensorfile + file_extension, header=0, names=sensor_index)
            #df_sensordata.set_index(sensor_index)

            # print Data frame
            #display(df)

            # plt.figure(figsize=(10,6))
            # plt.scatter(x, y)
            # plt.ylim(0,1000)
            # plt.show()


            #save file
            df_writedata = df_sensordata
            #add angle data to the last column
            df_writedata['node']=df_writedata['node'].mod(10) # incase when 2 expressed as 12
            df_writedata ['angle'] = 0.0
            df_writedata ['angle_velocity'] = 0.0
            #vertial node
            """ for i, row in df_writedata.iterrows():
                
                if (row['node'] == (3)):
                    xi = row['Bx']
                    intercept = intercept_V
                    coef = coef_V
                elif (row['node'] == (8)):
                    xi = row['Bx']
                    intercept = intercept_V
                    coef = coef_V
                else:  
                    xi = row['Bx']
                    intercept = intercept_H
                    coef = coef_H
                
                df_writedata.loc[i, 'angle'] = intercept +  \
                    coef[0]*xi**0 + coef[1]*xi**1 + coef[2]*xi**2 + \
                        coef[3]*xi**3 + coef[4]*xi**4 + coef[5]*xi**5
                print("calculate " + "i = " + str(i)) #+ " row = " + str(row)  ) 

            ## for faster iteration # pandas looping info 
            #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas?page=1&tab=scoredesc#tab-top
            #https://stackoverflow.com/questions/1422149/what-is-vectorization    
            #https://pythonspeed.com/articles/pandas-vectorization/
            #https://sparkbyexamples.com/pandas/iterate-over-rows-in-pandas-dataframe/#:~:text=itertuples()%20is%20the%20most,syntax%20of%20the%20itertuples()%20.&text=index%20%E2%80%93%20Defaults%20to%20%27True%27.

                """
            xi = df_writedata['Bx']
            intercept = intercept_H
            coef = coef_H

            df_writedata['angle'] = intercept_H +  \
                    coef_H[0]*df_writedata['Bx']**0 + coef_H[1]*df_writedata['Bx']**1 + \
                        coef_H[2]*df_writedata['Bx']**2 + coef_H[3]*df_writedata['Bx']**3 + \
                            coef_H[4]*df_writedata['Bx']**4 + coef_H[5]*df_writedata['Bx']**5

            df_writedata.loc[(df_writedata['node'] == (3)), 'angle'] = intercept_V +  \
                    coef_V[0]*df_writedata['Bx']**0 + coef_V[1]*df_writedata['Bx']**1 + \
                        coef_V[2]*df_writedata['Bx']**2 + coef_V[3]*df_writedata['Bx']**3 + \
                            coef_V[4]*df_writedata['Bx']**4 + coef_V[5]*df_writedata['Bx']**5

            df_writedata.loc[(df_writedata['node'] == (8)), 'angle'] = intercept_V +  \
                    coef_V[0]*df_writedata['Bx']**0 + coef_V[1]*df_writedata['Bx']**1 + \
                        coef_V[2]*df_writedata['Bx']**2 + coef_V[3]*df_writedata['Bx']**3 + \
                            coef_V[4]*df_writedata['Bx']**4 + coef_V[5]*df_writedata['Bx']**5

            df_writedata['angle'] = round(df_writedata['angle'], 2)
           
            ## angle masking
            df_writedata.loc[(df_writedata['angle'] < -90), 'angle'] = -90
            df_writedata.loc[(df_writedata['angle'] > 180), 'angle'] = 180
            ## remove corrupted data
            df_writedata_filtered = df_writedata[df_writedata['node'] > 0]

            # generate data folder if not exists
            if not os.path.exists(file_writefolder):
                os.mkdir(os.path.join(file_writefolder))
            
            # generate project folder if not exists
            if "kocca" in filename:
                folder_prj = "project_kocca"
            else:
                folder_prj = "project_kaist"

            if not os.path.exists(file_writefolder + file_startstring + folder_prj):
                os.mkdir(os.path.join(file_writefolder,str(folder_prj)))
            df_writedata_filtered.to_csv(file_writefolder + file_startstring + folder_prj +\
                 file_startstring + file_sensorfile + file_writestring + file_extension, sep=',', mode='w')
            
            if os.path.exists(file_clapfolder + file_startstring \
                    + file_sensorfile + file_clapstring + file_extension_clap):
                ## file split (on clap)
                # crop time frame data open
                clap_index = ['t_start', 't_end', 'shift']
                df_clap = pd.read_csv(file_clapfolder + file_startstring \
                    + file_sensorfile + file_clapstring + file_extension_clap, header=0, names=clap_index)
                
                #change timestring mm:ss to seconds
                
                df_start_s=pd.to_datetime(df_clap['t_start'], format='%M:%S').dt.second
                df_start_m=pd.to_datetime(df_clap['t_start'], format='%M:%S').dt.minute
                df_end_s=pd.to_datetime(df_clap['t_end'], format='%M:%S').dt.second
                df_end_m=pd.to_datetime(df_clap['t_end'], format='%M:%S').dt.minute
                df_clap['t_start'] = df_start_m*60 + df_start_s + df_clap['shift']
                df_clap['t_end'] = df_end_m*60 + df_end_s + df_clap['shift']

                # generate clop time folder if not exists
                if not os.path.exists(file_writefolder + file_startstring + folder_prj + file_startstring + "croptime"):
                    os.mkdir(os.path.join(file_writefolder + file_startstring + folder_prj,str("croptime")))
                df_clap.to_csv(file_writefolder + file_startstring + folder_prj + file_startstring + "croptime" \
                            + file_startstring + file_sensorfile + file_clapstring + file_extension, sep=',', mode='a')
                
            
                # generate croped sensor data folder if not exists
                if not os.path.exists(file_writefolder + file_startstring + folder_prj + file_startstring + "cropbyclap"):
                    os.mkdir(os.path.join(file_writefolder + file_startstring + folder_prj,str("cropbyclap")))
                
                for idx, row in df_clap.iterrows():
                    df_cropdata = df_writedata[(df_writedata['time'] >= row['t_start'])
                    & (df_writedata['time'] <= row['t_end'])]
                    df_cropdata.to_csv(file_writefolder + file_startstring + folder_prj + file_startstring + "cropbyclap" \
                            + file_startstring + file_sensorfile + file_writestring + "_rep_" + str(idx)\
                                + file_extension,sep=',', mode='a')
            
            # txt contents for CT
            # movement//joints//angle/angle_velocity
    # 1.233.219.136:9201

import socket
import glob


def send_line_by_line(file_path, sock):
    with open(file_path) as f:
        text = f.readlines()

        for line in text:
            sock.send(line.encode())

def send_all(file_path, sock):
    with open(file_path) as f:
        text = f.read()
        sock.send(text.encode())

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

file_path = glob.glob('dataprocessed/**/*.txt')[-1]
print(file_path)
sock.connect(('1.233.219.136', 9202))

send_all(file_path=file_path, sock=sock)