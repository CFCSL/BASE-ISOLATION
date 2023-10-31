# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:45:14 2023

@author: cfcpc2
"""

import pandas as pd
import numpy as np
from Base_Isolation_Calculation import AASHTO
import streamlit as st
import helper_functions as hf
from logo_header import *
import matplotlib.pyplot as plt
from Response_Spectrum import *


header()

logo()

st.markdown('---')
st.title("Base Isolation")

st.header("Samples")

st.sidebar.markdown("Parameters:")

m=st.sidebar.number_input("Number of supports $m$: ",min_value=1, value=4)
n=st.sidebar.number_input("Number of girder per support $n$: ", value=6, min_value=1)
n_col=st.sidebar.number_input("Number of columns per support $n_c$: ", value=3, min_value=1)

n_c=[0,n_col,n_col,0]

angle_skew=st.sidebar.number_input("Angle of skew $\\alpha$: ", value=0.0, max_value=np.pi/2, format="%.1f")
W=[]
W_default=[44.95, 280.31, 280.31, 44.95]
for i in range(int(m)):
	W.append(st.sidebar.number_input(f"Weight of superstructure at support {i+1}:",value=W_default[i], min_value=0.0, format="%.2f"))

W_PP=st.sidebar.number_input("Participating weight of piers $W_{PP}$ [k]",value=107.16, min_value=0.0, format="%.2f")
W_SS=np.sum(W)
W_eff=W_SS+W_PP

# Stiffness longitudinal
K_sub=[]
K_sub_default=[10000.0, 172.0, 172.0,10000.0]
for i in range(m):
	K_sub.append(st.sidebar.number_input(f"Stiffness of each pier in the longitudinal direction {i+1}:", value=K_sub_default[i], min_value=0.0, format="%.1f"))

# Stiffness transversal
K_sub1=[]
K_sub1_default=[10000.0, 687.0, 687.0,10000.0]
for i in range(m):
	K_sub1.append(st.sidebar.number_input(f"Stiffness of each pier in the trnsverse direction {i+1}:", value=K_sub1_default[i], min_value=0.0, format="%.1f"))

Isolator_Types=["Lead-rubber bearing","Spherical friction bearing","EradiQuake bearing"]
Isolator_Type=st.sidebar.selectbox("Isolator type", options=Isolator_Types)

PGA= st.sidebar.number_input("Peak Ground Acceleration $PGA$: ", value=0.4, min_value=0.0, format="%.2f")
S_1=st.sidebar.number_input("Long-Period Range of Spectrum Acceleration $S_1$: ", value=0.2, min_value=0.0,format="%.2f" )
S_S=st.sidebar.number_input("Short-Period Range of Spectrum Acceleration $S_S$: ", value=0.75, min_value=0.0,format="%.2f")

SiteClass_options=["B","A","C","D","E"]
SiteClass=st.sidebar.selectbox("SiteClass",options=SiteClass_options)

#######################################

#Samples=["Sample 1.0","Sample 1.1", "Sample 1.2","Sample 1.3","Sample 1.4","Sample 1.5","Sample 1.6","Sample 2.0","Sample 2.1", "Sample 2.2","Sample 2.3","Sample 2.4","Sample 2.5","Sample 2.6",]
st.markdown('---')
st.write('Generate Example')

#Sample= st.selectbox("Select the sample", options= Samples)

#q=st.sidebar.number_input("$q= $", value=0.05,min_value=0.0, format="%.2f")
#k=st.sidebar.number_input("$k= $", value=0.05,min_value=0.0, format="%.2f")
tol=st.number_input("Convergence check $tol=$",value= 0.05,min_value=0.0, format="%.3f")
T_max=st.number_input("Maximum period $T_{max}=$", value=2.0, format="%.2f")

##################

data=B1(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,T_max, Isolator_Type,tol,latex_format=False,plot_action=True)

df_conv=list(data.values())[-1] # Obtain the df from the last iteration

st.write(df_conv)

# Add a button to generate the plot
if st.button("Generate Plot"):
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
hf.download_csv(df_conv,file_name="Sample_1")




