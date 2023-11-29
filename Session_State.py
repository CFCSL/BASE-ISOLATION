# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:25:19 2023

@author: cfcpc2
"""
import streamlit as st

def Func_SessionState(a, b):
	if a is not None and b is not None:
		c = a * b
		return c
	else:
		return None



p1 = [1.0, 2.0]
p2 = [4.0, 5.0]
default=[None, None]


samples = {
	"manual input": default,
		'sample1_0': p1,
		'sample1_1': p2
	
}

selected_key = st.radio("**SAMPLES:**", options=list(samples.keys()), index=0, key="select_sample")
st.write(selected_key)

if st.button("Clear Parameters"):
	selected_key=list(samples.keys())[0]
	a = st.number_input("a=", format="%.2f", value= samples[selected_key][0])
	b = st.number_input("b=", format="%.2f", value= samples[selected_key][1])
else:
	# Display input fields for parameters
	a = st.number_input("a=", format="%.2f", value= samples[selected_key][0])
	b = st.number_input("b=", format="%.2f", value= samples[selected_key][1])

# Create a button to clear input parameters


if st.button("GENERATE", key="GENERATE"):
	
	c = Func_SessionState(a, b)
	st.write(f"The value is {c}")






