import asyncio # for concurrent (async/sync)
import numpy as np
import time as t
import struct
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures



from file_saving import *
#from Ai_recommend import *
from velocity_angleshift import *
from main_functions import *
from nicegui_graph import *

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

import requests
import pandas as pd
from nicegui import Tailwind, app, ui, events
from pathlib import Path
from pandas.api.types import is_bool_dtype, is_numeric_dtype
#from UI_back_main import *
# menu

Init_csvfolder()
csv_filename = ""

start_t = t.time()



# We need an executor to schedule CPU-intensive tasks with `loop.run_in_executor()`.
process_pool_executor = concurrent.futures.ProcessPoolExecutor()

open_Serial()    

serData_tmp = SerialData()
serData_tmp.lData = []

serData_tot = SerialData()
serData_tot.lData = [] 

serdata_rec = SerialData()
serData_rec = []


global ui_serialDf
global ui_serialDftemp
ui_serialDftemp = []
ui_serialDf = []

global df_sensor_data_all 

Recording_stat = False

serData_tot.lData = make_sensormagData_df([])
serdata_rec.lData = make_sensormagData_df([])
def readMagRealTime_init():
    
    t_interval = 0.5 
        
    t_end = time.time() + t_interval
    while time.time() < t_end:
        try:
                        
            cur_pkt_No = 0
            
            #serData_tot.lMagData.append(serData_tmp.lMagData)
            
            serialString = read_Serial()
            #print(serialString)
            
            if not serialString.empty:
                #thread pool (split 작업을 thread로 보냄)
                with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
                    
                        future = executor.submit(thrdf_serial_delimeter, serialString)
                        delimeted_serial = future.result() #serial output

                cur_pkt_No = delimeted_serial[15][0]

                with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
                    if cur_pkt_No > 0 : 
                        future = executor.submit(thrdf_get_Bmag, delimeted_serial)
                        cur_Bmagdata= future.result()
                        cur_Bmagdata.columns = ['Bx', 'By', 'Bz']
                        global serData_tmp
                        global serData_tot
                        
                        
                        serData_tmp.lData = cur_Bmagdata                                  
            
                        serData_tmp.decrypt()

                        serData_tmp.lData = add_time_col(serData_tmp.lData, cur_time(start_t))

                        serData_tmp.lData = add_node_col(serData_tmp.lData, cur_pkt_No)                       
                        serData_tot.lData = accumulate_sensor_Data(serData_tot.lData, serData_tmp.lData)
                        #header 없이 입력됨
                        global ui_serialDf
                        global ui_serialDftemp
                        ui_serialDf = serData_tot.lData
                        ui_serialDftemp = serData_tmp.lData
                        #print(serData_tmp.lData)
                        

                        ### no file saving (non-real time function)

        except: SyntaxError or AttributeError #when serial not connected

dtimefile = time.strftime("%Y%m%d-%H%M%S")

def file_write(dfname, filename, recoflag):
    
    if recoflag:
        save_path = file_writefolder + dtimefile + "_" + filename+file_extension
        outfile = open(save_path, 'wb')
        dfname.to_csv(outfile, sep=',', mode='a', header=False)
        outfile.close()
#df_rec = pd.DataFrame()
  
def readMagRealTime():
    #t_interval = 1
    #t_end = time.time() + t_interval
    #while time.time() < t_end:
    try:
                    
        cur_pkt_No = 0
        
        #serData_tot.lMagData.append(serData_tmp.lMagData)
        
        serialString = read_Serial()
        print(serialString)
        
        if not serialString.empty:
            #thread pool (split 작업을 thread로 보냄)
            with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
                
                    future = executor.submit(thrdf_serial_delimeter, serialString)
                    delimeted_serial = future.result() #serial output

            cur_pkt_No = delimeted_serial[15][0]

            with ThreadPoolExecutor(max_workers=NUM_SENSOR) as executor: 
                if cur_pkt_No > 0 : 
                    future = executor.submit(thrdf_get_Bmag, delimeted_serial)
                    cur_Bmagdata= future.result()
                    cur_Bmagdata.columns = ['Bx', 'By', 'Bz']
                    serData_tmp.lData = cur_Bmagdata                                  
        
                    serData_tmp.decrypt()

                    serData_tmp.lData = add_time_col(serData_tmp.lData, cur_time(start_t))

                    serData_tmp.lData = add_node_col(serData_tmp.lData, cur_pkt_No)     
                    
                    serData_tot.lData = accumulate_sensor_Data(serData_tot.lData, serData_tmp.lData)
                    #header 없이 입력됨


                    global ui_serialDf
                    global ui_serialDftemp
                    ui_serialDf = serData_tot.lData
                    ui_serialDftemp = serData_tmp.lData
                    print(serData_tmp.lData)
                    
                    
                    serdata_rec.lData = accumulate_sensor_Data(serdata_rec.lData, serData_tmp.lData)

            

        Recording_stat = Recocheckbox.value
        if Recording_stat:
            file_write(serdata_rec.lData, ui_input.value+"_"+ui_inputth.value, Recocheckbox.value)
        else:
            serdata_rec.lData = serData_tmp.lData


        
                                
    except: SyntaxError or AttributeError #when serial not connected


        


def update_serialdataframe(*, df: pd.DataFrame, r: int, c: int, value):
    df.iat[r, c] = value
    ui.notify(f'Set ({r}, {c}) to {value}')

## Front end

readMagRealTime_init()

with ui.row().classes('w-full items-center'):
    result = ui.label().classes('mr-auto')
    with ui.button(icon='menu'):
        with ui.menu() as menu:
            ui.menu_item('Intro', lambda: result.set_text('Intro'))
            ui.menu_item('Serial', lambda: result.set_text('Serial'))
            ui.menu_item('Graph', lambda: result.set_text('Graph'))
            ui.menu_item('Content', lambda: result.set_text('Content'))
            ui.separator()
            ui.menu_item('Close', on_click=menu.close)

ui.image('./Images/Muments_logo.jpg').classes('w-40 h-40').classes('items-center')


ui_input = ui.input(label='Text', placeholder='start typing',
         on_change=lambda e: {result.set_text('you typed: ' + e.value)},
         validation={'Input too long': lambda value: len(value) < 100})
ui.label().bind_text_from(ui_input, 'value')

ui_inputth = ui.input(label='th Test', placeholder='start typing',
         on_change=lambda e: {result.set_text('you typed: ' + e.value)},
         validation={'Input too long': lambda value: len(value) < 100})
ui.label().bind_text_from(ui_input, 'value')

count = 0

def Toggle_func(cur_recordst):

    Recocheckbox.value = cur_recordst

    Recordstring =""
    if Recording_stat:
        Recordstring = " Now Recording @ " + csv_filename + ".csv"
    else:
        Recordstring = "Saved File"
    ui.notify(Recordstring)

class ToggleButton(ui.button):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state = False
        self.on('click', self.toggle)
               
    def toggle(self) -> None:
        """Toggle the button state."""
        self._state = not self._state        
        Toggle_func(self._state)
        self.update()

    def update(self) -> None:
        self.props(f'color={"red" if self._state else "green"}')
        super().update()      

ToggleButton('Record').classes('rounded-full w-16 h-16 ml-4')

Recocheckbox = ui.checkbox('Recording check')
ui.label('Checked!').bind_visibility_from(Recocheckbox, 'value')



with ui.expansion('Graph',  value=True, icon='done').classes('w-full items-center'):
    ui.image('./Images/240321_sensor_position.png').classes('w-40 h-40').classes('items-center')
    
    graph_wdth = 5
    graph_hght = 2
    up_interval = 2

             
    with ui.grid(columns=3):

        line_plot1 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)
        line_plot1.classes('w-full items-center')

        ui.label("Shoulder").classes('w-full items-center').style('font-size: 200%; font-weight: 500').tailwind.font_weight('extrabold')

        line_plot6 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3) 
        line_plot6.classes('w-full items-center')

    
    with ui.grid(columns=3):

        line_plot2 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)
        
        ui.label("Elbow").classes('w-full items-center').style('font-size: 200%; font-weight: 500').tailwind.font_weight('extrabold')
       
        line_plot7 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)
        
    
    with ui.grid(columns=3):

        line_plot3 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)     

        ui.label("Hip").classes('w-full items-center').style('font-size: 200%; font-weight: 500').tailwind.font_weight('extrabold')
        
        line_plot8 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3) 
        
    
    with ui.grid(columns=3):

        line_plot4 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)
        
        ui.label("Knee").classes('w-full items-center').style('font-size: 200%; font-weight: 500').tailwind.font_weight('extrabold')

        line_plot9 = ui.line_plot(n=3, limit=20, figsize=(graph_wdth, graph_hght), update_every=up_interval) \
            .with_legend(['Bx', 'By', 'Bz'], loc='upper right', ncol=3)

    def update_line_plot() -> None:
        #now = datetime.now()
        readMagRealTime()
        #t = ui_serialDftemp['time'].values.tolist()
        #y1 = ui_serialDftemp['Bx'].values.tolist()
        #y2 = ui_serialDftemp['By'].values.tolist()
        #y3 = ui_serialDftemp['Bz'].values.tolist()

        t = ui_serialDftemp['time'][0]
        y1 = ui_serialDftemp['Bx'][0]
        y2 = ui_serialDftemp['By'][0]
        y3 = ui_serialDftemp['Bz'][0]
        node = ui_serialDftemp['node'][0]

        if node == 1:
            line_plot1.push([t], [[y1], [y2], [y3]])
        if node == 2:
            line_plot2.push([t], [[y1], [y2], [y3]])
        if node == 3:
            line_plot3.push([t], [[y1], [y2], [y3]])
        if node == 4:
            line_plot4.push([t], [[y1], [y2], [y3]])
        if node == 6:
            line_plot6.push([t], [[y1], [y2], [y3]])
        if node == 7:
            line_plot7.push([t], [[y1], [y2], [y3]])                                
        if node == 8:
            line_plot8.push([t], [[y1], [y2], [y3]])
        if node == 9:
            line_plot9.push([t], [[y1], [y2], [y3]])
        
    line_updates = ui.timer(1, update_line_plot, active=True)
    line_checkbox = ui.checkbox('active').bind_value(line_updates, 'active')

    #with splitter.after:       

            


ui.label('STAY IN THE PRESENT MOMENT 2023').classes('absolute-bottom text-center')
#ui.run(port=9000)
ui.run(reload = False, port=6777)








