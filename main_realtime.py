import asyncio # for concurrent (async/sync)
import numpy as np
import time as t
import struct
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
#import nicegui.globals


from file_saving import *
#from Ai_recommend import *
from velocity_angleshift import *
from main_functions import *

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

#UI
ui.add_head_html('''
<style>
@font-face {
    font-family: 'PretendardJP-ExtraBold';
    src: url('./fonts/PretendardJP-ExtraBold.woff') format('woff');
}

// declare a class which applies it
.my-font {
  font-family: 'PretendardJP-ExtraBold';
  
</style>
''')

app.add_static_file(local_file='./fonts/PretendardJP-ExtraBold.woff', url_path='./PretendardJP-Bold.woff')


#Global constants
""" NUM_SENSOR = 10
CONVERSION_FACTOR = 10 # for conversion
MAX_STACK_SIZE = 10
MAX_packet_dataset = 5 # of Bx, By, Bz set
T_send = 500 #ms
CHAR_TERMINATION = b'\r\n'
COM_PORT = 'COM5' # set to 0 when not used """
        

start_t = t.time()
infoRecoYoutube = []


# We need an executor to schedule CPU-intensive tasks with `loop.run_in_executor()`.
process_pool_executor = concurrent.futures.ProcessPoolExecutor()
make_csv()
open_Serial()    

serData_tmp = SerialData()
serData_tmp.lData = []

serData_tot = SerialData()
serData_tot.lData = []    

global df_sensor_data_all 

serData_tot.lData = make_sensormagData_df([])



t_end = time.time() + 10#seconds 
while time.time() < t_end:
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
                    
                    ui_serialDf = serData_tot.lData
                    print(serData_tmp.lData)

    except KeyboardInterrupt:
        pass
    
write_csv_filename(serData_tot.lData, "Pretest")
close_Serial()

mvel, mshift = vel_ang_shift(serData_tot.lData)



if not (mvel or mshift):
    mvel = 1
    mshift = 1    

print("mean velocity:" + str(mvel))
print("mean shift:" + str(mshift))
""" 
small_no = 50
score_happy = 1/abs(mshift) + abs(mvel) + small_no
score_anger =  1/abs(mshift)  + 1/abs(mvel) + small_no
score_fear =  abs(mshift)  + 1/abs(mvel) + small_no
score_sad =  abs(mshift)  + abs(mvel) + small_no
  """
""" score_happy = 0.4 * abs(mshift) + 0.6 * abs(mvel)
score_sad =  0.3 * abs(mshift)  + 0.7 * abs(mvel)
score_fear =  0.7 * abs(mshift)  + 0.3 * abs(mvel)
score_anger =  0.4 * abs(mshift)  + 0.6 * abs(mvel) """


#max_score = max([abs(score_happy),abs(score_sad),abs(score_fear),abs(score_anger)])



#emotionscore=np.array([score_happy,score_sad,score_fear,score_anger]).reshape(1,-1) #Emotion vector

#infoRecoYoutube, RecoYoutubeResults, RecoMsg, Emotionresult= recommendSong(emotionscore, 1)
#infoRecoYoutubeId = get_YoutubeInfo(infoRecoYoutube)
#print(infoRecoYoutubeId)

def update_serialdataframe(*, df: pd.DataFrame, r: int, c: int, value):
    df.iat[r, c] = value
    ui.notify(f'Set ({r}, {c}) to {value}')


#ui_serialDf = pd.DataFrame([1,2,3,4,5,6,7,8,9,10, 11, 12, 13,14,15,16])
#ui_serialDf = thread_serialreaderdf(read_Serial())

#ui.image('./Images/Upper_bar_bg.jpg').classes('h-20 items-end')
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

#topVidUrl = './media/top_perform_vid.mp4'
#topVidUrl = 'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4'
""" with ui.video(src=topVidUrl, muted=True, loop=True, autoplay=True).classes('w-full items-center'):
    ui.label('Nice!').classes('absolute-bottom text-subtitle2 text-center color-white') """

#with ui.splitter().classes('w-full items-center') as splitter:
    #with splitter.before:
    #Current serial
    
with ui.expansion('Serial Data', icon='done').classes('w-full items-center'):
    ui.label('contents')
    with ui.grid(rows=len(ui_serialDf .index)+1).classes('grid-flow-col'):
        for c, col in enumerate(ui_serialDf.columns):
            ui.label(col).classes('font-bold')
            for r, row in enumerate(ui_serialDf .loc[:, col]):
                if is_bool_dtype(ui_serialDf [col].dtype):
                    cls = ui.checkbox
                elif is_numeric_dtype(ui_serialDf [col].dtype):
                        cls = ui.number
                else:
                    cls = ui.input
                cls(value=row, on_change=lambda event, r=r, c=c: update_serialdataframe(df=ui_serialDf, r=r, c=c, value=event.value))

#Current Graph
with ui.expansion('Graph', icon='done').classes('w-full items-center'):
    ui.label('contents')
    #with splitter.after:       
    
""" with ui.expansion('Emotion ', icon='done').classes('w-full items-center'):
    emoString = '아마도 이건 그런 '+str(Emotionresult)+'한 기분'
    ui.label(emoString).classes('text-center').style('font-family: PretendardJP-ExtraBold').style('font-size: 200%; font-weight: 300').tailwind.font_weight('extrabold')
    #infoRecoYoutube, RecoYoutubeResults, RecoMsg, Emotionresult """
    
""" with ui.expansion('Muments recommends you', value=True, icon='done').classes('w-full items-center'):
    ui.label('Youtube #'+infoRecoYoutubeId).classes('text-center').style('font-size: 200%; font-weight: 300').tailwind.font_weight('extrabold')
    ui.label(RecoMsg).classes('text-center').style('font-size: 200%; font-weight: 300').tailwind.font_weight('extrabold')
    recmmendYoutubeID = infoRecoYoutubeId
    
    iframePrefix = '<div width="100%" align="center"><iframe id="ytplayer" width="420" height="315" src="https://www.youtube.com/embed/'
    iframePostfix = '?&autoplay=1&mute=0&cc_lang_pref=fr&cc_load_policy=0;start=60";rel=0;modestbranding frameborder="0" allowfullscreen></iframe></div>'
    ui.html(iframePrefix+recmmendYoutubeID+iframePostfix) """
            


ui.label('STAY IN THE PRESENT MOMENT 2023').classes('absolute-bottom text-center')
#thread_serialreaderdf()


ui.run(port=9000)








