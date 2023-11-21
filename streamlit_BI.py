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


#########################################
#Samples=["Sample 1.0","Sample 1.1", "Sample 1.2","Sample 1.3","Sample 1.4","Sample 1.5","Sample 1.6","Sample 2.0","Sample 2.1", "Sample 2.2","Sample 2.3","Sample 2.4","Sample 2.5","Sample 2.6",]
#st.markdown('---')
#st.write('Generate Example')

#Sample= st.selectbox("Select the sample", options= Samples)

#q=st.number_input("$q= $", value=0.05,min_value=0.0, format="%.2f")
#k=st.number_input("$k= $", value=0.05,min_value=0.0, format="%.2f")
tol=st.number_input("Convergence check $tol=$",value= 0.05,min_value=0.0, format="%.3f")
T_max=st.number_input("Maximum period $T_{max}=$", value=2.0, format="%.2f")

##################
st.markdown('---')
st.markdown("**The final result occurs when the problem converges**")
params=m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol

#st.write(f'sample1_0={params}')


# =============================================================================
# 
# data=B1(params,latex_format=False,plot_action=True)
# 
# df_conv=list(data.values())[-1] # Obtain the df from the last iteration
# 
# st.write(df_conv)
# 
# # Add a button to generate the plot
# if st.button("**Generate Plot**"):
# 	t = np.linspace(0, T_max,200)
# 	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 
# 
# 	# Plot the design response spectrum
# 	fig, ax = plt.subplots()
# 	ax.plot(t, C_sm)
# 	ax.set_title(f"Design Response Spectrum for PGA={PGA}, S_S={S_S}, S_1={S_1}, SiteClass={SiteClass}")
# 	ax.set_xlabel('Period')
# 	ax.set_ylabel('Acceleration')
# 
# 	# Display the plot in the Streamlit app
# 	st.pyplot(fig)
# 	
# =============================================================================
	
#params=m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol	
params1_0=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05
params1_1=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"D",2.0,"Lead-rubber bearing", 0.075,0.1,0.05
params1_2=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.6,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05
params1_3=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Spherical friction bearing", 0.05,0.05,0.05
params1_4=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"EradiQuake bearing", 0.05,0.05,0.05
params1_5=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.05,0.05,0.05
params1_6=4,6,[0,3,3,0],650.52,107.16,[44.95,280.31,280.31,44.95],[10000.0,172.0,172.0,10000.0],0.0,0.40,0.2,0.75,"B",2.0,"Lead-rubber bearing", 0.075,0.1,0.05


# Define the sample parameters
samples = {
    'sample1_0': params1_0,
    'sample1_1': params1_1,
    'sample1_2': params1_2,
    'sample1_3': params1_3,
    'sample1_4': params1_4,
    'sample1_5': params1_5,
    'sample1_6': params1_6,
}

# Display the sample selection dropdown in the sidebar
selected_sample = st.sidebar.selectbox("Select the sample:", options=list(samples.keys()),key="select sample")

# Automatically fill out the input fields based on the selected sample
if selected_sample in samples:
    st.sidebar.markdown("**Sample Parameters:**")
    m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol = samples[selected_sample]
    st.sidebar.number_input("Number of supports $m$:", min_value=1, value=m)
    st.sidebar.number_input("Number of girder per support $n$:", min_value=1, value=n)
    st.sidebar.number_input("Number of columns per support $n_c$:", min_value=1, value=n_c[1])
    st.sidebar.number_input("Angle of skew $\\alpha$:", value=angle_skew, max_value=np.pi/2, format="%.1f")
    st.sidebar.selectbox("Isolator type", options=Isolator_Types, index=Isolator_Types.index(Isolator_Type))
    st.sidebar.number_input("Peak Ground Acceleration $PGA$:", min_value=0.0, value=PGA, format="%.2f")
    st.sidebar.number_input("Long-Period Range of Spectrum Acceleration $S_1$:", min_value=0.0, value=S_1, format="%.2f")
    st.sidebar.number_input("Short-Period Range of Spectrum Acceleration $S_S$:", min_value=0.0, value=S_S, format="%.2f")
    st.sidebar.text_input("SiteClass", options=SiteClass_options, index=SiteClass_options.index(SiteClass))
    st.sidebar.number_input('q=', min_value=0.0, value=q, format="%.3f")
    st.sidebar.number_input('k=', min_value=0.0, value=k, format="%.3f")
    st.sidebar.number_input("Participating weight of piers $W_{PP}$:", min_value=0.0, value=W_PP, format="%.2f")
    st.sidebar.number_input("Convergence check $tol$:", min_value=0.0, value=tol, format="%.3f")
    st.sidebar.number_input("Maximum period $T_{max}$:", value=T_max, format="%.2f")













#st.write(f'sample1_1={params1_1}')
option = st.sidebar.selectbox("Select the sample:", options=['sample1_0', 'sample1_1','sample1_2', 'sample1_3','sample1_4', 'sample1_5','sample1_6'])
if option=='sample1_0':
	data = B1(params, latex_format=False, plot_action=True)
if option=='sample1_1':
	data = B1(params1_1, latex_format=False, plot_action=True)
if option=='sample1_2':
	data = B1(params1_2, latex_format=False, plot_action=True)
if option=='sample1_3':
	data = B1(params1_3, latex_format=False, plot_action=True)
if option=='sample1_4':
	data = B1(params1_4, latex_format=False, plot_action=True)
if option=='sample1_5':
	data = B1(params1_5, latex_format=False, plot_action=True)
if option=='sample1_6':
	data = B1(params1_6, latex_format=False, plot_action=True)


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

#
# Download CSV
#hf.download_csv(df_conv,file_name="Sample_1")




