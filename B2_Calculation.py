# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:18:07 2023

@author: cfcpc2
"""
import pandas as pd 
from Response_Spectrum import AASHTO
import matplotlib.pyplot as plt
import numpy as np
from Base_Isolation_Calculation import *


def B2(params,T_eff, latex_format=True,plot_action=False):
	m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol = params


# Calculate Response Spectrum:
	# Create a array of time:
	shape=200
	t=np.linspace(0,T_max,shape)
	# Call the Response Spectrum function
	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 


## Step 1: Recall the function B1
	dt=B1(params,d=2,latex_format=False,plot_action=False)
	data=list(dt.values())[-1]
	display(data)
	d=data.d.unique()[0]
	d_new=data.d_new.unique()[0]
	Q_d=data.Q_d.unique()[0]
	K_d=data.K_d.unique()[0]
	alpha_j=data.alpha_j.to_list()
	K_effj=data.K_effj.to_list()
	d_isolj=data.d_isolj.to_list()
	Q_dj=data.Q_dj.to_list()
	#print(Q_dj)
	K_dj=data.K_dj.to_list()
	#print(K_dj)
	K_isolj=data.K_isolj. to_list()
	#print(K_isolj)
	#T_eff=data.T_eff.unique()[0]
	K_eff=data.K_eff.unique()[0]
	xi=data.xi.unique()[0]
	B_L=data.B_L.unique()[0]
	d_subj=data.d_subj.to_list()



## Step 2: Multimode  
	"""
	- The results from the Simplified Method (Step B1 in Article B2.1.2.1)
	are used to determine initial values for the equivalent spring elements for the isolators as a starting point in the
	iterative process. 
   
	"""

	
	
	### Begin iteration
	iter_multimode =1



	##Step B2.1:Characteristic Strength

	##%% Calculate the characteristic strength, Qd,i, and postelastic stiffness, Kd,i, of each isolator “i” 
	
	Q_di=[Q_dj[j]/n for j in range(m)]
	
	K_di=[K_dj[j]/n for j in range(m)]
	
	#print(f'K_di={K_di}')

	##Step B2.2: Initial Stiffness and Yield Displacement

	##%% Calculate the initial stiffness, Ku,i, and the yield displacement, dy,i, for each isolator “i”

	if Isolator_Type == "friction-based isolators":
		K_ui =[np.inf for j in range(m)] # n=6, is the numbers of isolator per each support 
		d_yi=[0]*m
	else:
		K_ui=[10*K_di[j] for j in range(m)]
		d_yi=[Q_di[j]/(K_ui[j]-K_di[j]) for j in range(m)]
	#print(f'K_ui={K_ui}')
	#print(f'd_yi={d_yi}')
	#while True:	
	##Step B2.3: Isolator Effective Stiffness, Kisol,i

	##%% Calculate the isolator stiffness, Kisol,i, of each isolator “i”
	
	
	k_isoli=[K_isolj[j]/n for j in range (m)] #(sample: n=6, m=4)
		
	#print(f'k_isoli={k_isoli}')
	
	##Step B2.4: ThreeDimensional Bridge Model
	
	##Step B2.5: Composite Design Response Spectrum
	if plot_action==True:
		N=int((0.8*T_eff*shape)/T_max)
		C_sm[N:]=C_sm[N:]/B_L # C_sm is calculated by Response Spectrum, and B_L is the result from the last  convergence step.
		## Show the plot here
		plt.plot(t, C_sm)
		plt.title(f"Design Response Spectrum Multi Modes for PGA={PGA}, S_S={S_S}, S_1= {S_1}, SiteClass={SiteClass}")
		plt.xlabel(f'Period')
		plt.ylabel(f'Acceleration')
		plt.show()

	##Step B2.6: Multimode Analysis of Finite Element Model

		#( Call the d_isolj from the previuos calculation)

		##%% Recalculate system damping ratio, ξ :
		
		d_1= 9.79*S_D1*T_eff/B_L
		print(f'd_new={d_1}')
		# Update d_soli 
		dt1=B1(params,d=d_1,latex_format=False,plot_action=False)
		data1=list(dt1.values())[-1]
		display(data1)
		d_isoli=data1.d_isolj.to_list()
		print(f'd_isoli={type(d_isoli)}')
		#d_new=data1.d_new.unique()[0]
		#d1=df.d_new.unique()[0]
		#d_subj=df.d_subj.to_list()
		
		#print(d_new)

		tol=abs((d_new-d)/d)
		
		if tol<=0.02:
			print('convergence checked')
		else:
			print('continue iteration')
			
# =============================================================================
# 				
# 				
# 			## B2.1.2.2.8—Step B2.8: Update Kisol,i, Keff,j, ξ , and BL
# 		
# 			##%% Use the calculated displacements in each isolator element to obtain new values of Kisol,i for each isolator
# 		
# 			K_isoli=[Q_di[j]/d_isoli[j]+K_di[j] for j in range(m)] # (d_isoli in the document) is d_isolj in the convergence step
# 			
# 			print(f'K_isoli={K_isoli}')
# 			###%% Recalculate K_effj
# 		
# 			K_effj=[(K_sub[j]*K_isoli[j]*n)/(K_sub[j]+K_isoli[j]*n) for j in range(m)]
# 			
# 			print(f'K_effj={K_effj}')
# 		
# 		
# 	
# 			
# 		
# 		
# 		
# 	# =============================================================================
# 	# 	
# 	# 	##%% Calculate alpha_j
# 	# 
# 	# 	alpha_i = [(K_dj[i]*d_multimode+Q_dj[i])/(K_sub[i]*d_multimode-Q_dj[i]) for i in range(m)]
# 	# 
# 	# 	##%% Calculate the displacement of the isolation system, d_isolj
# 	# 
# 	# 	d_isoli=  [d_multimode/(1+ alpha_i[i]) for i in range(m)]
# 	# 
# 	# =============================================================================
# 			#print(f'd_isoli={d_isoli}')
# 			numerator=2*sum([n*Q_di[j]*(d_isoli[j]-d_yi[j]) for j in range(m)])
# 			denominator=np.pi*sum([K_effj[j]*(d_isoli[j]+d_subj[j])**2 for j in range(m)]) 
# 		
# 			xi=numerator/denominator
# 			#print(f'xi={xi}')
# 			
# 			## Recalculate system damping factor, BL:
# 			#B_L=np.piecewise(((xi/0.05)**0.3,xi<0.3),(1.7, xi>=0.3))
# 			B_L=np.piecewise(xi, [xi<0.3,xi>=0.3], [(xi/0.05)**0.3,1.7])
# 			#print(f'B_L={B_L}')
# 			#Recalculate Keff
# 		
# 			K_eff=sum(K_effj)
# 			
# 			#print(f'K_eff={K_eff}')
# 			
# 			#print(f'W_eff {W_eff}')
# 			
# 			#print(f'g={g}')
# 		
# 			# Recalculate T_eff
# 			W_eff= W_SS + W_PP #  Effective weight, W_eff
# 			g=386.4 # (in./s^2) or 9.815(m/s^2)
# 			T_eff= 2*np.pi*(W_eff/(g*K_eff))**(1/2)
# 			
# 			#print(f'T_eff={T_eff}')
# 		
# 			## Recalculate d
# 			d_multimode_new=(9.79*S_D1*T_eff)/B_L
# 			#print(f'd_multimode_new={d_multimode_new}')
# 	
# 			d=d_multimode_new
# 			iter_multimode+=1
# 
# 	print(f'd_multimode_new {round(d_multimode_new,2)}')
# 
# 	print(f'Numbers of iteratations to get convergence check: {iter_multimode}')
# 	
# 	
# 	#superstructure displacements in the longitudinal (xL) and transverse (yL) directions are:
# 
# 	x_L=d_isoli[0]*np.cos(angle_skew)
# 	y_L=d_isoli[0]*np.sin(angle_skew)
# 	print(f'Superstructure displacements in the longitudinal:\n x_L={round(x_L,2)}\n y_L={round(y_L,2)}')
# 
# 	# isolator displacements in the longitudinal (uL) and transverse (vL) directions are:
# 	u_L=[]
# 	v_L=[]
# 	# Abutment:
# 	u_L.append(d_isoli[0]*np.cos(angle_skew))
# 	v_L.append(d_isoli[0]*np.sin(angle_skew))
# 	print(f'isolator displacements in the longitudinal (uL) and transverse (vL) directions are:')
# 	print(f'Abutments: u_L={round(u_L[0],2)}, v_L={round(v_L[0],2)}')
# 
# 	#Piers:
# 
# 	u_L.append(d_isoli[1]*np.cos(angle_skew))
# 	v_L.append(d_isoli[1]*np.sin(angle_skew))
# 	print(f'Piers: u_L={round(u_L[1],2)}, v_L={round(v_L[1],2)}')
# 
# 
# 
# 
# 
# =============================================================================


# =============================================================================
# 
# 	
# 	df_multi = pd.DataFrame({"Pier": ["Abut1", "Pier1", "Pier2", "Abut2"],
# 							"$$Q_{di}$$": Q_di,
# 							"$$K_{di}$$": K_di,
# 							"$$K_{ui}$$":K_ui,
# 							"$$d_{yj}$$":d_yj,
# 							"$$k_{isol,i}$$":k_isoli,
# 							"$$K_{isol,i}$$": K_isoli,
# 							"$\\alpha_i$": alpha_i, 
# 							"$$K_{eff,j}$$": K_effj,
# 							"$$\\xi$$":xi,
# 							"$$B_{L}$$": B_L,
# 							"$$K_{eff}$$":K_eff,
# 							"$$T_{eff}$$": T_eff,
# 							"d_multimode_new":d_multimode_new
# 							})
# 	data_multimode[iter_multimode]=df_multi
# 	
# 	for k in data_multimode.keys():
# 		for col in data_multimode[k].columns:
# 			if col in ["$\\alpha_j$", "$$d_{sub,j}$$"]:
# 				data_multimode[k][col]=data_multimode[k][col].apply(scientific_format)
# 			else:
# 				data_multimode[k][col]=data_multimode[k][col]. apply(round_values, n=2)
# 
# 	if latex_format==False:
# 		for i in df_multi.keys():
# 			df_multi[i]=df_multi[i].rename(columns={
# 					"$$Q_{di}$$": "Q_di",
# 					"$$K_{di}$$": "K_di",
# 					"$$K_{ui}$$":"K_ui",
# 					"$$d_{yj}$$":"d_yj",
# 					"$$k_{isol,i}$$":"k_isoli",
# 					"$$K_{isol,i}$$": "K_isoli",
# 					"$\\alpha_i$": "alpha_i", 
# 					"$$K_{eff,j}$$": "K_effj",
# 					"$$\\xi$$":"xi",
# 					"$$B_{L}$$": "B_L",
# 					"$$K_{eff}$$":"K_eff"
# 				})
# 
# 
# 
# 
# 
# 
# 	return data_multimode
# 		
# 
# =============================================================================
