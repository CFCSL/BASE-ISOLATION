# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:45:14 2023

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



m=st.number_input("Number of supports $m$: ",min_value=1, value=4)
n=st.number_input("Number of girder per support $n$: ", value=6, min_value=1)
n_col=st.number_input("Number of columns per support $n_c$: ", value=3, min_value=1)

n_c=[0,n_col,n_col,0]

angle_skew=st.number_input("Angle of skew $\\alpha$: ", value=0.0, max_value=np.pi/2, format="%.1f")

Isolator_Types=["Lead-rubber bearing","Spherical friction bearing","EradiQuake bearing"]
Isolator_Type=st.selectbox("Isolator type", options=Isolator_Types, key="isolator_type_selectbox")

PGA= st.number_input("Peak Ground Acceleration $PGA$: ", value=0.4, min_value=0.0, format="%.2f")
S_1=st.number_input("Long-Period Range of Spectrum Acceleration $S_1$: ", value=0.2, min_value=0.0,format="%.2f" )
S_S=st.number_input("Short-Period Range of Spectrum Acceleration $S_S$: ", value=0.75, min_value=0.0,format="%.2f")

SiteClass_options=["B","A","C","D","E"]
SiteClass=st.selectbox("SiteClass",options=SiteClass_options)

q=st.number_input('q=', value=0.05, max_value=1.0,format="%.3f") # 5%
k=st.number_input('k=', value=0.05, max_value=1.0,format="%.3f") # 5%

W_PP=st.number_input("Participating weight of piers $W_{PP}$",value=107.16, min_value=0.0, format="%.2f")

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
dt = {
 "Weight (W_j)": [44.95, 280.31, 280.31, 44.95],
 "Longitudinal stiffness (K_subj)": [10000.0, 172.0, 172.0, 10000.0],
 "Transverse stiffness": [10000.0, 687.0, 687.0, 10000.0]
}
index_values = ['Abutment 1', "Pier 1", "Pier 2", 'Abutment 2']

df = pd.DataFrame(dt,index=index_values)
df=st.data_editor(df, num_rows= "dynamic")
df=df.dropna(how="all", axis=0)



W=df["Weight (W_j)"].to_list()

K_sub=df["Longitudinal stiffness (K_subj)"].to_list()
K_sub1=df["Transverse stiffness"].to_list()

W_SS=np.sum(W)
W_eff= W_SS+np.sum(W_PP)

tol=st.number_input("Convergence check $tol=$",value= 0.05,min_value=0.0, format="%.3f")
T_max=st.number_input("Maximum period $T_{max}=$", value=2.0, format="%.2f")


default_params=[m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol]
data = B1(default_params, latex_format=False, plot_action=True)
df_conv = list(data.values())[-1]
st.write(df_conv)

# Add a button to generate the plot
#if st.sidebar.button("**Generate Plot**"):
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
##################
st.markdown('---')


#st.markdown("**The final result occurs when the problem converges**")

# parameters for each sample:
params1_0=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],np.pi/4,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05]
params1_1=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.3,"D",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]
params1_2=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.6,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]
params1_3=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Spherical friction bearing", 0.05,0.05,0.05]
params1_4=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"EradiQuake bearing", 0.05,0.05,0.05]
params1_5=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05]
params1_6=[4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05]

# Define the sample parameters
samples = {
#	'default':default_params,
	'sample1_0': params1_0,
	'sample1_1': params1_1,
	'sample1_2': params1_2,
	'sample1_3': params1_3,
	'sample1_4': params1_4,
	'sample1_5': params1_5,
	'sample1_6': params1_6,
}

# Display the sample selection dropdown in the sidebar
#selected_sample = st.sidebar.selectbox("Select the sample:", options=list(samples.keys()),key="select sample")
# =============================================================================
# bt=st.button("SAMPLES",key="btSamples")
# if bt:
# =============================================================================
selected_sample = st.sidebar.radio("**SAMPLES:**", options=list(samples.keys()), index=0, key="select_sample")
# Automatically fill out the input fields based on the selected sample
params=default_params
data = B1(params, latex_format=False, plot_action=True)
if selected_sample in samples:
	
	params=samples[selected_sample]
	params_dict = {
		'm': params[0],
		'n': params[1],
		'n_c': params[2],
		'W_SS': params[3],
		'W_PP': params[4],
		'W': params[5],
		'K_sub': params[6],
		'angle_skew': params[7],
		'PGA': params[8],
		'S_1': params[9],
		'S_S': params[10],
		'SiteClass': params[11],
		'T_max': params[12],
		'Isolator_Type': params[13],
		'q': params[14],
		'k': params[15],
		'tol': params[16]
		}
	# Set values in Streamlit app using sidebar
	m = st.sidebar.number_input("Number of supports $m$:", value=params_dict['m'], min_value=1)
	n = st.sidebar.number_input("Number of girder per support $n$:", value=params_dict['n'], min_value=1)
	n_col = st.sidebar.number_input("Number of columns per support $n_c$:", value=params_dict['n_c'][1], min_value=1)
	angle_skew = st.sidebar.number_input("Angle of skew $\\alpha$:", value=params_dict['angle_skew'], max_value=np.pi/2, format="%.1f")
	Isolator_Type=st.sidebar.text_input("Isolator type", value=params_dict['Isolator_Type'])
	PGA= st.sidebar.number_input("Peak Ground Acceleration $PGA$: ", value=params_dict['PGA'])
	S_1=st.sidebar.number_input("Long-Period Range of Spectrum Acceleration $S_1$: ", value=params_dict['S_1'] )
	S_S=st.sidebar.number_input("Short-Period Range of Spectrum Acceleration $S_S$: ", value=params_dict['S_S'])
	SiteClass = st.sidebar.text_input("SiteClass",value=params_dict['SiteClass'])
	q=st.sidebar.number_input('q=', value=params_dict['q'], max_value=1.0) # 5%
	k=st.sidebar.number_input('k=', value=params_dict['k'], max_value=1.0) # 5%

	W_PP=st.sidebar.number_input("Participating weight of piers $W_{PP}$",value=params_dict['W_PP'],)
	tol=st.sidebar.number_input("Convergence check $tol=$",value= params_dict['tol'])
	
	[m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol] = params
	data = B1(params, latex_format=False, plot_action=True)
	
	df_conv = list(data.values())[-1]
	st.sidebar.write(df_conv)

	# Add a button to generate the plot
	#if st.sidebar.button("**Generate Plot**"):
	t = np.linspace(0, T_max,200)
	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 

	# Plot the design response spectrum
	fig, ax = plt.subplots()
	ax.plot(t, C_sm)
	ax.set_title(f"Design Response Spectrum for PGA={PGA}, S_S={S_S}, S_1={S_1}, SiteClass={SiteClass}")
	ax.set_xlabel('Period')
	ax.set_ylabel('Acceleration')

	# Display the plot in the Streamlit app
	st.sidebar.pyplot(fig)







#
# Download CSV
#hf.download_csv(df_conv,file_name="Sample_1")

