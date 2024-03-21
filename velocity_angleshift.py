import pandas as pd

#unsined_cov = 1000

def vel_ang_shift(df_sensor):
    df_node_write = df_sensor.drop_duplicates(subset='time')

    df_t_1 = df_node_write['time'][0:-1].reset_index(drop=True)
    df_t_2 = df_node_write['time'][1:].reset_index(drop=True)
    df_Bx_1 = df_node_write['Bx'][0:-1].reset_index(drop=True)
    df_Bx_2 = df_node_write['Bx'][1:].reset_index(drop=True)
    df_By_1 = df_node_write['By'][0:-1].reset_index(drop=True)
    df_By_2 = df_node_write['By'][1:].reset_index(drop=True)
    df_Bz_1 = df_node_write['Bz'][0:-1].reset_index(drop=True)
    df_Bz_2 = df_node_write['Bz'][1:].reset_index(drop=True)
    
    df_vel_Bx = (df_Bx_2 - df_Bx_1) / ( df_t_2 - df_t_1 )
    df_vel_By = (df_By_2 - df_By_1) / ( df_t_2 - df_t_1 )
    df_vel_Bz = (df_Bz_2 - df_Bz_1) / ( df_t_2 - df_t_1 )
    
    
    df_dBx = (df_Bx_2 - df_Bx_1)
    #df_time_shift = ( df_t_2 - df_t_1 ) # check (debug var)
    df_dBy = (df_By_2 - df_By_1)
    #df_time_shift = ( df_t_2 - df_t_1 ) # check (debug var)
    df_dBz = (df_Bz_2 - df_Bz_1)
    #df_time_shift = ( df_t_2 - df_t_1 ) # check (debug var)


    mean_vel = (df_vel_Bx.mean() + df_vel_By.mean() + df_vel_Bz.mean())/3
    mean_ang = ((df_dBx).mean() + df_dBy.mean() + df_dBz.mean())/3

    return mean_vel, mean_ang