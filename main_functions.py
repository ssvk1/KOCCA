
import serial
import asyncio
import serial.tools.list_ports as port_list
import numpy as np
import time as t
import struct
from nicegui import app, ui, events
# for data postprocessing
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os # for rawdata loading
from io import StringIO
import pandas as pd

file_writefolder = "csv_result/"
file_name = "realtime"
file_extension = ".txt"

def make_csv():
    for file in os.listdir(file_writefolder):
                
        filename, ext = os.path.splitext(file)

        # generate data folder if not exists
        if not os.path.exists(file_writefolder):
            os.mkdir(os.path.join(file_writefolder))


def write_csv(df_write):
        df_write.to_csv(file_writefolder + file_name+file_extension, sep=',', mode='a', header=False)

process_pool_executor = concurrent.futures.ProcessPoolExecutor()


#Global constants
NUM_SENSOR = 8
CONVERSION_FACTOR = 10 # for conversion
MAX_STACK_SIZE = 10
MAX_packet_dataset = 5 # of Bx, By, Bz set
T_send = 500 #ms
CHAR_TERMINATION = b'\r\n'
COM_PORT = 0 #'COM30' # set to 0 when not used

ports = list(port_list.comports())

serialPort = serial.Serial()
serialString = ""
port=COM_PORT

start_t = t.time()
      
class SerialData():
    def __init__(self):
        lData = pd.DataFrame([])
        
    def add(self, arr):
        np.append(arr, self.lData)

    def decrypt(self):
        self.lData = self.lData/CONVERSION_FACTOR


def make_sensor_df(df_new_sensor_arr):

    sensor_index = ['time','node','Bx','By','Bz','|B|mag','Angle', 'Angle_vel']
    df_new_sensor_arr = pd.DataFrame(df_new_sensor_arr, columns=sensor_index)
    return df_new_sensor_arr

def make_sensormagData_df(df_sensor_Mag_arr):
    sensor_index = ['Bx','By','Bz']
    df_new_sensor_arr  = pd.DataFrame(df_sensor_Mag_arr, columns=sensor_index)
    return df_new_sensor_arr

def accumulate_sensor_Data(df_sensor_arr, df_sensor_added_arr):
                                  
    df_sensordata = pd.concat([df_sensor_arr, df_sensor_added_arr])
    return df_sensordata

def add_time_col(df_sensor_arr, in_time):
    df_sensor_arr['time'] = in_time
    t_interval = 0.01
    df_sensor_arr['time'][0] = in_time
    df_sensor_arr['time'][1] += t_interval*1
    df_sensor_arr['time'][2] += t_interval*2
    df_sensor_arr['time'][3] += t_interval*3
    df_sensor_arr['time'][4] += t_interval*4
    return df_sensor_arr


def add_node_col(df_sensor_arr, cur_node):
    df_sensor_arr['node'] = cur_node
    return df_sensor_arr


serData_tmp = SerialData()
serData_tmp.lData = []

serData_tot = SerialData()
serData_tot.lData = []    

#df_sensor_data_all 
#infoRecoYoutubeId
#infoRecoYoutube

def isSerialAvailable(serData):
    isavailable = False
    numreadbytes = serData.in_wating

    if numreadbytes:
        isavailable = True

def read_Serial():
    if COM_PORT:
        serialString = serialPort.read_until(CHAR_TERMINATION)
        serialString = serialString.rstrip(CHAR_TERMINATION)
    else: 
        predefinedString = "390, -590, -320, 380, -590, -330, 380, -590, -330, 380, -590, -340, 390, -580, -340, 3"
        serialString = predefinedString
    if __debug__:
        print(serialString)
    
    serialStringDf = pd.DataFrame(eval(serialString)).T
    
    return serialStringDf


def open_Serial():
    if ports:
        if COM_PORT:
            port = COM_PORT
        else:
            print(ports[0].device)
            port = ports[0].device
        print(port)
        baudrate = 115200
        global serialPort
        serialPort = serial.Serial(port=port, baudrate=baudrate,\
            bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE)\
                
def close_Serial():
    serialPort.close()
                



def thread_serialreader() -> serialString:
    loop = asyncio.get_running_loop()
    serialString = loop.run_in_executor(None, read_Serial)
    
    return serialString

def thread_serialreaderdf():
    #loop = asyncio.get_running_loop()
    serialString = read_Serial()
    if serialString:
        #thread pool (split 작업을 thread로 보냄)
        with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
            
                future = executor.submit(thrdf_serial_delimeter, serialString)
                delimeted_serial = future.result() #serial output

        cur_pkt_No = delimeted_serial[15]

        with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
            if cur_pkt_No > 0 : 
                future = executor.submit(thrdf_get_Bmag, delimeted_serial)
                cur_Bmagdata= future.result()
                serData_tmp.lData = make_sensormagData_df(cur_Bmagdata)

                serData_tmp.decrypt()

                serData_tmp.lData = add_time_col(serData_tmp.lData, cur_time(start_t))

                serData_tmp.lData = add_node_col(serData_tmp.lData, cur_pkt_No)

                serData_tot.lData = accumulate_sensor_Data(serData_tot.lData, serData_tmp.lData)
                ui_serialDf = serData_tot.lData
                
                #header 없이 입력됨
                write_csv(serData_tmp.lData)
                print(serData_tmp.lData)
    
    return ui_serialDf 

""" async def disconnect() -> None:
    Disconnect all clients from current running server.
    for client in nicegui.globals.clients.keys():
        await app.sio.disconnect(client) """

async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()
    # Release the webcam hardware so it can be used by other applications again.
    read_Serial.release()
    # The process pool executor must be shutdown when the app is closed, otherwise the process will not exit.
    process_pool_executor.shutdown()




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

           
def cur_time(s_time):
    time_c = t.time()- s_time
    return time_c
        
def thrdf_serial_delimeter(in_string): #thrdf : thread function

    if isinstance(in_string, pd.DataFrame):
        new_str = in_string
    else: #mostly list[]
        new_str = in_string.split(b",")
        for i in range(len(new_str)):
            #new_str[i] = int.from_bytes(new_str[i].decode()) # xx
            new_str[i] = int(new_str[i].decode()) 
    
    return new_str

def thrdf_get_Bmag(in_string_arr):
    if isinstance(in_string_arr, pd.DataFrame):
        new_string_arr = pd.DataFrame()
        print(in_string_arr)
        new_string_arr = \
            pd.DataFrame(in_string_arr.iloc[0:, :15].values.reshape(-1,3))
        """ new_string_arr = in_string_arr.iloc[:, 0:3]
        print(new_string_arr)
        new_string_arr = pd.concat([new_string_arr, in_string_arr.iloc[:, 3:6]]) 
        print(new_string_arr)
        new_string_arr = pd.concat([new_string_arr, in_string_arr.iloc[:, 6:9]]) 
        print(new_string_arr)
        new_string_arr = pd.concat([new_string_arr, in_string_arr.iloc[:, 9:12]]) 
        print(new_string_arr)
        new_string_arr = pd.concat([new_string_arr, in_string_arr.iloc[:, 12:15]]) 
        print("new strin arr") """
        print(new_string_arr)
        
    else:
        new_string_arr = in_string_arr[0:15]
        new_string_arr = list(new_string_arr)
    return new_string_arr

""" 
if __name__ == '__main__':
    if __debug__:
        pass """


    