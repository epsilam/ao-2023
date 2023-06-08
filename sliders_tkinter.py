# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:36:56 2023

@author: Group 2
"""

###############################################################################
# Running this script starts a window with sliders to adjust the actuators    #
# of the OKOTech deformable mirror.                                           #
###############################################################################


import numpy as np
import matplotlib.pyplot as plt

import time
import os

import tkinter as tk

lb_act = -0.4 # actuator lower bound
ub_act = 0.3  # actuator upper bound
init_act = 0.18

def get_dm_position_from_slider():
    values = [slider.get() for slider in actuator_sliders]
    return(np.asarray(values))
    
# update function to set DM coefficients when a slider for an individual actuator is touched
def update_on_slider_change(val):
    # the function must have an argument val but we ignore it
    time.sleep(0.1)
    dm_position = get_dm_position_from_slider()
    dm.setActuators(dm_position)
    
# update function to set all DM coefficients at once with the single uniform slider
def update_uniform(val):
    
    # Visually update all the individual sliders
    for var in slider_value_vars:
        var.set(uniform_slider.get())
    
    time.sleep(0.1)
    dm_position = uniform_slider.get()*np.ones(len(dm))
    dm.setActuators(dm_position)
    

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=0) as dm:
        
        # Initialize tkinter master object
        master = tk.Tk()
        
        # Canvas and scrollbar configuration
        main_frame = tk.Frame(master)
        main_frame.pack(fill="both",expand=1)        
        canvas = tk.Canvas(main_frame)
        canvas.pack(side="left", fill="both", expand=1)
        sb = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        sb.pack(side="right",fill="y")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        frame_sliders = tk.Frame(canvas)
        canvas.create_window((0,0), window=frame_sliders, anchor="nw")
        
        # Create sliders for each individual actuator
        actuator_sliders = []
        slider_value_vars = []
        for actuator_idx in range(len(dm)):
            slider_value_var = tk.DoubleVar()
            slider = tk.Scale(frame_sliders, 
                         from_=lb_act, to=ub_act,
                         resolution=0.01,
                         length=300,
                         orient="horizontal",
                         label=f'Actuator {actuator_idx}',
                         command=update_on_slider_change,
                         variable=slider_value_var)
            slider.grid(row=actuator_idx + 1,column=0)
            slider.set(init_act)
            actuator_sliders.append(slider)
            slider_value_vars.append(slider_value_var)
        
        # Create a slider which sets all actuators to the same value when it is moved
        uniform_slider = tk.Scale(frame_sliders, 
                     from_=lb_act, to=ub_act,
                     resolution=0.01,
                     length=300,
                     orient="horizontal",
                     label=f'Set all actuators equally',
                     command=update_uniform)
        uniform_slider.grid(row=0,column=0)
            
        tk.mainloop()