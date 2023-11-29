# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:02:20 2023

@author: cfcpc2
"""

import pandas as pd
import numpy as np
from Base_Isolation_Calculation import B1
import streamlit as st
import helper_functions as hf
from logo_header import *
import matplotlib.pyplot as plt
from Response_Spectrum import AASHTO


header()

logo()

st.markdown('---')
st.title("SEISMIC ISOLATION DESIGN")

st.header('Manual input')
st.markdown("**Parameters:**")


st.markdown('---')




# parameters for each sample:

params1_0=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05]
params1_1=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"D",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]
params1_2=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.6,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]
params1_3=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Spherical friction bearing", 0.05,0.05,0.05]
params1_4=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"EradiQuake bearing", 0.05,0.05,0.05]
params1_5=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05]
params1_6=[4,6,[0,3,3,0],107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]

# Define the sample parameters
samples = {

	'Sample1_0': params1_0,
	'Sample1_1': params1_1,
	'Sample1_2': params1_2,
	'Sample1_3': params1_3,
	'Sample1_4': params1_4,
	'Sample1_5': params1_5,
	'Sample1_6': params1_6,
}

selected_sample = st.radio("**SAMPLES:**", options=list(samples.keys()), index=0, key="select_sample")
# Automatically fill out the input fields based on the selected sample

if selected_sample in samples:
	params=samples[selected_sample]
	# default_params=[m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol]
	params_dict = {
		'm': params[0],
		'n': params[1],
		'n_c': params[2],
		'W_PP': params[3],
		'W': params[4],
		'K_sub': params[5],
		'angle_skew': params[6],
		'PGA': params[7],
		'S_1': params[8],
		'S_S': params[9],
		'SiteClass': params[10],
		'T_max': params[11],
		'Isolator_Type': params[12],
		'q': params[13],
		'k': params[14],
		'tol': params[15]
		}
	# Set values in Streamlit app using sidebar
	m = st.number_input("Number of supports $m$:", value=params_dict['m'], min_value=1)
	n = st.number_input("Number of girder per support $n$:", value=params_dict['n'], min_value=1)
	n_c = st.number_input("Number of columns per support $n_c$:", value=params_dict['n_c'][1], min_value=1)
	n_col=[0,n_c,n_c,0]
	angle_skew = st.number_input("Angle of skew $\\alpha$:", value=params_dict['angle_skew'], max_value=np.pi/2, format="%.1f")
	Isolator_Type=st.text_input("Isolator type", value=params_dict['Isolator_Type'])
	PGA= st.number_input("Peak Ground Acceleration $PGA$: ", value=params_dict['PGA'])
	S_1=st.number_input("Long-Period Range of Spectrum Acceleration $S_1$: ", value=params_dict['S_1'] )
	S_S=st.number_input("Short-Period Range of Spectrum Acceleration $S_S$: ", value=params_dict['S_S'])
	SiteClass = st.text_input("SiteClass",value=params_dict['SiteClass'])
	q=st.number_input('q=', value=params_dict['q'], max_value=1.0) # 5%
	k=st.number_input('k=', value=params_dict['k'], max_value=1.0) # 5%

	W_PP=st.number_input("Participating weight of piers $W_{PP}$",value=params_dict['W_PP'])
	st.write("**$W$ and $K_{sub}$**")

	dt = {
	  "W": params_dict['W'],
	  "K_sub":  params_dict['K_sub'],
	  "Trans.stiffness": [10000.0, 687.0, 687.0, 10000.0]
	 }
	index_values = ['Abutment 1', "Pier 1", "Pier 2", 'Abutment 2']
	 
	df = pd.DataFrame(dt,index=index_values)
	df=st.data_editor(df, num_rows= "dynamic")
	df=df.dropna(how="all", axis=0)
	st.markdown("""

	- Weight of superstructure, $$W_j$$, at each support.
	
	- Stiffness, $$K_{subj}$$, of each support in both 
	longitudinal and transverse directions of the 
	bridge. The calculation of these quantities 
	requires careful consideration of several factors 
	such as the use of cracked sections when 
	estimating column or wall flexural stiffness, 
	foundation flexibility, and effective column 
	height.
	""")

	W=df["W"].to_list()
	K_sub=df["K_sub"].to_list()
	K_sub1=df["Trans.stiffness"].to_list()
	tol=st.number_input("Convergence check $tol=$",value= params_dict['tol'])
	T_max=st.number_input("Maximum period $T_{max}=$", value=params_dict['T_max'], format="%.2f")
	
	st.markdown('---')
	
	st.markdown("**The final result occurs when the problem converges**")
	p=[m, n, n_col, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol] 
	data = B1(p, latex_format=False, plot_action=True)
	
	## Show the plot
	t = np.linspace(0, T_max,200)
	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 

	# Plot the design response spectrum
	fig, ax = plt.subplots()
	ax.plot(t, C_sm)
	ax.set_title(f"Design Response Spectrum for PGA={PGA}, S_S={S_S}, S_1={S_1}, SiteClass={SiteClass}")
	ax.set_xlabel('Period')
	ax.set_ylabel('Acceleration')

	# Display the plot in the Streamlit app
	st.pyplot(fig)

	df_conv = list(data.values())[-1]
	st.write(df_conv)



