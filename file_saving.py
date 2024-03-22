import os # for rawdata loading
from io import StringIO
import pandas as pd
import time

file_writefolder = "csv_result/"
file_name = ""
file_extension = ".txt"
dtimenow = time.strftime("%Y%m%d-%H%M%S")

def update_dtime ():
    dtimenow = time.strftime("%Y%m%d-%H%M%S")

def get_dtime():
    return time.strftime("%Y%m%d-%H%M%S")

def Init_csvfolder():
    for file in os.listdir(file_writefolder):
                
        filename, ext = os.path.splitext(file)

        # generate data folder if not exists
        if not os.path.exists(file_writefolder):
            os.mkdir(os.path.join(file_writefolder))



def write_csv(df_write):
        df_write.to_csv(file_writefolder + dtimenow + "_" + file_name+file_extension, sep=',', mode='a', header=False)

def write_csv_filename(df_write, file_name):
        df_write.to_csv(file_writefolder + dtimenow + "_" + file_name+file_extension, sep=',', mode='a', header=False)

def open_csv():
    outfile = open(file_writefolder + dtimenow + "_" + file_name+file_extension, 'wb')
    return outfile

def write_csv_fileopen(df_write, in_filename, fileflag):
    if fileflag:
        with open(file_writefolder + dtimenow + "_" + in_filename+file_extension, "w") as filehandle:
            df_write.to_csv(filehandle, sep=',', mode='a', header=False)
    else:
        filehandle.close()
        pass

def close_csv(filehandle):
    filehandle.close()