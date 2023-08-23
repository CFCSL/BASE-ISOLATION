#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:20:10 2023

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



st.header("AASHTO")




st.write("**Peak Ground Acceleration:**")
PGA=st.number_input("PGA",value= 0.1, min_value=0.0, step=0.05, format="%.3f")

st.write("**Short-Period Range of Spectrum Acceleration:**")
S_S=st.number_input("$S_S$",value= 0.25, min_value=0.0, step=0.05, format="%.3f")

st.write("**Long-Period Range of Spectrum Acceleration:**")
S_1=st.number_input("$S_1$",value= 0.1, min_value=0.0, step=0.05, format="%.3f")



st.write("**Site Class**: ")
options=["A","B","C","D","E","F"]
default_options = ["A","B","C","D","E"]  # Set all options as default
SiteClass=st.multiselect("Select options:", options, default=default_options)

if "F" in SiteClass:
	st.warning("Site-specific geotechnical investigation and dynamic site response analysis should be performed for all sites in Site Class F")
	
try:
	SiteClass.remove("F")
except:
	pass


st.write("**Period T[s]**")

T_max=st.number_input("Select T_max[s]", value= 6., min_value=1., step=1., format="%.3f")
x = np.linspace(0.01,T_max , 200)


# Create the plot
fig, ax = plt.subplots()
for k in SiteClass:

    ax.plot(x, RS.AASHTO(x, PGA=PGA, S_S=S_S, S_1=S_1, SiteClass=k), label=k)

    ax.legend()
    #ax.set_title("Design Response Spectrum"+ "PGA=",PGA, + "$S_S$=", S_S, + "$S_1$=", S_1, + "Site Class=", SiteClass)
    ax.set_title("Design Response Spectrum PGA={}, $S_S$={}, $S_1$={}, Site Class={}".format(PGA, S_S, S_1, SiteClass))
    
# Set the x-axis and y-axis labels
ax.set_xlabel('T[s]')
ax.set_ylabel('$C_{sm}$')

# Display the plot in Streamlit
st.pyplot(fig)



# Create an empty dataframe
df = pd.DataFrame({'Period[s]': x})
#df1=pd.DataFrame({'Frequency[1/s]': (1/x)})
list_df=[]
list_df1=[]
# create interations
for k in SiteClass:
	df_k = pd.DataFrame({'Period[s]': x,
                         "C_sm"+"-"+str(k): RS.AASHTO(x, PGA=PGA, S_S=S_S, S_1=S_1, SiteClass=k)})
	df1_k = pd.DataFrame({'Frequency[1/s]': (1/x),
                         "C_sm"+"-"+str(k): RS.AASHTO(x, PGA=PGA, S_S=S_S, S_1=S_1, SiteClass=k)})
	# sort column 'Frequency[1/s]' in ascending order
	df1_k=df1_k.sort_values('Frequency[1/s]').round(4)

	# Merge df and df_k on the "Frequency[1/s]" column
	df = pd.merge(df, df_k, on="Period[s]")

	# Append the df_k into the list
	list_df.append(df_k)
	list_df1.append(df1_k)

df=df.round(4)

st.write("")
st.write(df)

# Download CSV
hf.download_csv(df,file_name="AASHTO")
hf.download_sofistik(list_df,file_name="AASHTO")
hf.download_abaqus(list_df1,file_name="AASHTO")


