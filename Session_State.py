# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:25:19 2023

@author: cfcpc2
"""
import streamlit as st
import pandas as pd
#from FuncSessionState import *

# =============================================================================
# 
# # Initialization
# if 'key' not in st.session_state:
#     st.session_state['key'] = 'value'
# 
# st.write(st.session_state.key)
# 
# 
# df=pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
# 
# st.session_state['fixed_table']=df
# 
# df1=st.data_editor(df,key=f'editor_1')
# 
# 
# 
# st.markdown('---')
# 
# st.write(st.session_state)
# 
# st.markdown('---')
# 
# st.write(df1)
# 
# st.markdown('---')
# 
# st.write(st.session_state[f'editor_1'])
# 
# =============================================================================
def Func_SessionState(*params):
	a, b = params
	c = a * b
	return c


p1 = [1, 2]
p2 = [4, 5]

samples = {
	'sample1_0': p1,
	'sample1_1': p2
}

selected_key = st.radio("**SAMPLES:**", options=list(samples.keys()), index=0, key="select_sample")

st.write(selected_key)



a= st.number_input("a=", format="%.2f",value=samples[selected_key][0] )
b= st.number_input("b=", format="%.2f",value=samples[selected_key][1] )

#c=Func_SessionState(samples[selected_key][0],samples[selected_key][1])
c=Func_SessionState(a,b)
st.write(f" the value is {c}")















# =============================================================================
# 
# selected_key = st.sidebar.radio("**SAMPLES:**", options=list(samples.keys()), index=0, key="select_sample")
# if selected_key in samples:
# 	params = samples[selected_key]
# 	params_dict = {
# 		'a': params[0],
# 		'b': params[1]
# 	}
# 
# 	# Convert the scalar values into a list to create a DataFrame
# 	df = pd.DataFrame([params_dict])
# 
# 	df=st.data_editor(df)
# 	p=[]
# 	for k in df:
# 		p.append(df[k])
# 	
# 	c1 = Func_SessionState(p)
# 	st.write(f" the value is {c1[0]}")
# =============================================================================

