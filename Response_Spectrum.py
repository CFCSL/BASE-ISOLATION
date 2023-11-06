#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:40:50 2023

@author: namnguyen
"""
## import Library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from ipywidgets import interact, interactive, fixed, interact_manual
#import ipywidgets as widgets


#%% EuroCode8

def EC8(T,ag=0.5,GroundType='A',Eta=1,Dir='Horizontal',RS_Type=1):
	#Turn it into Array:
	T=np.atleast_1d(T)
	T=T.astype('float64')
	Ground={1:{},2:{}}
	#Ground[RS_Type][GroundType]=[S,TB,TC,TD]
	#Type 1
	Ground[1]['A']=[1.0, 0.15,0.4,2.0]
	Ground[1]['B']=[1.2, 0.15,0.5,2.0]
	Ground[1]['C']=[1.15,0.20,0.6,2.0]
	Ground[1]['D']=[1.35,0.20,0.8,2.0]
	Ground[1]['E']=[1.4, 0.15,0.5,2.0]
	#Type 2
	Ground[2]['A']=[1.0, 0.05,0.25,1.2]
	Ground[2]['B']=[1.35,0.05,0.25,1.2]
	Ground[2]['C']=[1.5, 0.10,0.25,1.2]
	Ground[2]['D']=[1.8, 0.10,0.30,1.2]
	Ground[2]['E']=[1.6, 0.05,0.25,1.2]
	# Dir= Vartical:
	Ground[1]['Vertical']=[0.90,1.0, 0.05,0.15,1.0]
	Ground[2]['Vertical']=[0.45,1.0, 0.05,0.15,1.0]
	if Dir=='Horizontal':
		S,TB,TC,TD=Ground[RS_Type][GroundType]
		aa=ag
		_dir_coeff=2.5
	elif Dir=='Vertical':
		avg_ratio,S,TB,TC,TD=Ground[RS_Type][Dir]
		aa=ag*avg_ratio
		_dir_coeff=3.0
	else:
		print('Error: set dir=Vertical or Horizontal')
	condList=[(0<=T)&(T<TB),(TB<=T)&(T<=TC),(TC<T)&(T<=TD),TD<T]
	funcList=[lambda T: aa*S*(1+(T/TB)*(Eta*_dir_coeff-1)),lambda T: aa*S*Eta*_dir_coeff,lambda T: aa*S*Eta*_dir_coeff*(TC/T),lambda T: aa*S*Eta*_dir_coeff*((TC*T)/T**2)]
	Amp=np.piecewise(T, condList, funcList)
	return Amp




#%% ASSHTO

def AASHTO(T, PGA,S_S,S_1,SiteClass): #col is position =0/1/2/3/4
	#Turn it into Array:
	T=np.atleast_1d(T)
	T=T.astype('float64')
	SiteFactor={'Zero_Period':{},'Short_Period':{}, 'Long_Period':{}}

	#SiteFactor[RS_Type][SiteClass]=[pos0,pos1,pos2,pos3,pos4]
	#F_pga
	SiteFactor['Zero_Period']['A']=[0.8,0.8,0.8,0.8,0.8]
	SiteFactor['Zero_Period']['B']=[1.0,1.0,1.0,1.0,1.0]
	SiteFactor['Zero_Period']['C']=[1.2,1.2,1.1,1.0,1.0]
	SiteFactor['Zero_Period']['D']=[1.6,1.4,1.2,1.1,1.0]
	SiteFactor['Zero_Period']['E']=[2.5,1.7,1.2,0.9,0.9]
	SiteFactor['Zero_Period']['F']=["*","*","*","*","*"]
	   
	#F_a
	SiteFactor['Short_Period']['A']=[0.8,0.8,0.8,0.8,0.8]
	SiteFactor['Short_Period']['B']=[1.0,1.0,1.0,1.0,1.0]
	SiteFactor['Short_Period']['C']=[1.2,1.2,1.1,1.0,1.0]
	SiteFactor['Short_Period']['D']=[1.6,1.4,1.2,1.1,1.0]
	SiteFactor['Short_Period']['E']=[2.5,1.7,1.2,0.9,0.9]
	SiteFactor['Short_Period']['F']=["*","*","*","*","*"]
	   
	#F_v
	SiteFactor['Long_Period']['A']=[0.8,0.8,0.8,0.8,0.8]
	SiteFactor['Long_Period']['B']=[1.0,1.0,1.0,1.0,1.0]
	SiteFactor['Long_Period']['C']=[1.7,1.6,1.5,1.4,1.3]
	SiteFactor['Long_Period']['D']=[2.4,2.0,1.8,1.6,1.5]
	SiteFactor['Long_Period']['E']=[3.5,3.2,2.8,2.4,2.4]
	SiteFactor['Long_Period']['F']=["*","*","*","*","*"]

	def f(x,a,b,Fa,Fb):
		return Fa+(Fb-Fa)/(b-a)*(x-a)

	def F_pga(PGA, SiteClass):
		condList = [PGA < 0.1,(PGA>=0.1)& (PGA <= 0.2), (PGA>0.2)&(PGA <= 0.3), (PGA>0.3)&(PGA <= 0.4),(PGA>0.4)&(PGA<=0.5), PGA > 0.5]
		F = SiteFactor['Zero_Period'][SiteClass]
		valList=[F[0],f(PGA,0.1,0.2,F[0],F[1]),f(PGA,0.2,0.3,F[1],F[2]),f(PGA,0.3,0.4,F[2],F[3]),f(PGA,0.4,0.5,F[3],F[4]),F[4]]
		F_pga = np.piecewise(PGA, condList, valList)
		return F_pga


	def F_a(S_S,SiteClass):
		condList=[S_S<0.25, (S_S>=0.25)&(S_S<=0.5),(S_S>=0.5)&(S_S<=0.75),(S_S>=0.75)&(S_S<=1.0),(S_S>=1)&(S_S<=1.25), S_S>=1.25]
		F = SiteFactor['Short_Period'][SiteClass]
		valList=[F[0],f(S_S,0.25,0.5,F[0],F[1]),f(S_S,0.5,0.75,F[1],F[2]),f(S_S,0.75,1.0,F[2],F[3]),f(S_S,1.0,1.25,F[3],F[4]),F[4]]
		F_a = np.piecewise(S_S, condList, valList)
		return F_a
		
	def F_v(S_1, SiteClass):
		condList = [S_1 < 0.1,(S_1>=0.1)& (S_1 <= 0.2), (S_1>0.2)&(S_1 <= 0.3), (S_1>0.3)&(S_1 <= 0.4),(S_1>0.4)&(S_1<=0.5), S_1 > 0.5]
		F = SiteFactor['Long_Period'][SiteClass]
		valList=[F[0],f(S_1,0.1,0.2,F[0],F[1]),f(S_1,0.2,0.3,F[1],F[2]),f(S_1,0.3,0.4,F[2],F[3]),f(S_1,0.4,0.5,F[3],F[4]),F[4]]
		F_v = np.piecewise(S_1, condList, valList)
		return F_v


	F_pga=F_pga(PGA,SiteClass)
	F_a=F_a(S_S,SiteClass)
	F_v=F_v(S_1,SiteClass)
	
	A_S=F_pga*PGA
	S_DS=F_a*S_S
	S_D1=F_v*S_1
	
	T_S=S_D1/S_DS
	T_0=0.2*T_S
	
	condList=[(0<=T)&(T<=T_0),(T_0<T)&(T<=T_S),(T_S<T)]
	funcList=[lambda T: A_S+(S_DS-A_S)*T/T_0,lambda T: S_DS,lambda T: S_D1/T]
	C_sm=np.piecewise(T, condList, funcList)
	
	return C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1

def Plot(T_max, PGA, S_S, S_1, SiteClass):    
    x = np.linspace(0.01, T_max, 200)
    C_sm, F_pga, F_a, F_v, A_S, S_DS, S_D1 = AASHTO(T=x, PGA=PGA, S_S=S_S, S_1=S_1, SiteClass=SiteClass)
    
    plt.plot(x, C_sm)

    # Set the title, x-axis label, and y-axis label
    plt.title(f"Design Response Spectrum for Site Class ={SiteClass}")
    plt.xlabel('T [s]')
    plt.ylabel('Csm')

    # Show the plot
    plt.show()








