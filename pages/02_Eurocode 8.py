#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:53:20 2023

@author: namnguyen
"""
import pandas as pd
import streamlit as st
import Response_Spectrum as RS
import helper_functions as hf
import numpy as np
import matplotlib.pyplot  as plt
import zipfile
from io import BytesIO
import base64


st.header("Eurocode 8")




st.write('**Direction:**')
Dir=st.radio("Select Direction:", ("Horizontal", "Vertical"))


st.write("**Response Spectrum:**")
RS_Type = st.radio("Response Spectrum Type", ["Type 1", "Type 2"], index=0)
RS_Type_value = 1 if RS_Type == "Type 1" else 2


st.write("G**round Type**: ")
options=["A","B","C","D","E"]
default_options = options  # Set all options as default
GroundType=st.multiselect("Select options:", options, default=default_options)


a_g=st.number_input("a_g [$m/s^2$]", value= 0.5, min_value=0.0, step=0.1, format="%.3f")

st.write("**Period T[s]**")


T_max=st.number_input("Select T_max[s]", value= 6., min_value=1., step=1., format="%.3f")
x = np.linspace(0.01,T_max , 200)

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(5, 8))

for k in GroundType:
    ax1.plot(x, 1/a_g*RS.EC8(x, GroundType=k, Dir=Dir, RS_Type=RS_Type_value), label=k)
    ax2.plot(x, RS.EC8(x, GroundType=k, Dir=Dir, RS_Type=RS_Type_value), label=k)

ax1.legend()
ax1.set_title(f"{Dir}-elastic response spectra")
ax1.set_xlabel('T[s]')
ax1.set_ylabel('$S_e/a_g$')

ax2.legend()
ax2.set_title(f"{Dir}-elastic response spectra of amplitude acceleration vs period")
ax2.set_xlabel('T[s]')
ax2.set_ylabel('$S_e$')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)





# Create an empty dataframe
df = pd.DataFrame({'Period[s]': x})
list_df=[]
list_df1=[]
# create interations
for k in GroundType:
	df_k=pd.DataFrame({'Period[s]':x,'S_e'+"-"+str(k):RS.EC8(x,GroundType=k,Dir=Dir,RS_Type=RS_Type_value)*9.81})
	df1_k=pd.DataFrame({'Frequency[1/s]':(1/x),'S_e'+" "+str(k):RS.EC8(x,GroundType=k,Dir=Dir,RS_Type=RS_Type_value)*9.81})
	# sort column 'Frequency[1/s]' in ascending order
	df1_k=df1_k.sort_values('Frequency[1/s]').round(4)
	
	# Merge df and df_k on the "Frequency[1/s]" column
	df = pd.merge(df, df_k, on="Period[s]")

	# Append the df_k into the list
	
	list_df.append(df_k)
	list_df1.append(df1_k)
	
	
df=df.round(4)

st.write(df)

# Download CSV
hf.download_csv(df,file_name="EC8")
hf.download_sofistik(list_df,file_name="EC8")
hf.download_abaqus(list_df1,file_name="EC8")









