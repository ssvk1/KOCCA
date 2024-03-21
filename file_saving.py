import os # for rawdata loading
from io import StringIO
import pandas as pd
import time

file_writefolder = "csv_result/"
file_name = ""
file_extension = ".txt"
dtimenow = time.strftime("%Y%m%d-%H%M%S")

def make_csv():
    for file in os.listdir(file_writefolder):
                
        filename, ext = os.path.splitext(file)

        # generate data folder if not exists
        if not os.path.exists(file_writefolder):
            os.mkdir(os.path.join(file_writefolder))


def write_csv(df_write):
        df_write.to_csv(file_writefolder + dtimenow + "_" + file_name+file_extension, sep=',', mode='a', header=False)

def write_csv_filename(df_write, file_name):
        df_write.to_csv(file_writefolder + dtimenow + "_" + file_name+file_extension, sep=',', mode='a', header=False)
