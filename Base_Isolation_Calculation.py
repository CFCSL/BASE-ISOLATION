# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
from Response_Spectrum import *
from sympy import symbols, Eq, Function,UnevaluatedExpr, Mul
from sympy import *
import matplotlib.pyplot as plt

#init_printing()
from sympy import Piecewise, nan
import numpy as np


#%%
def round_expr(expr, num_digits=2):
	return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

def round_equation(eq, num_digits=2):
	lhs = eq.lhs
	rhs = eq.rhs
	rounded_rhs = round_expr(rhs, num_digits)
	return Eq(lhs, rounded_rhs)

# Round the values in each column to n decimal places
def round_values(x,n):
	try:
		return round(x, n)
	except:
		return x
	
# Apply the e-4 formatting to numeric columns
def scientific_format(x):
	try:
		float_value = float(x)
		return '{:.2e}'.format(float_value).replace('+', '')
	except:
		return x


#%%
def B1(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,T_max, Isolator_Type,tol,latex_format=True,plot_action=False, d=2.0):
	
	"""
	m: Number of supports
	
	n: Number of girders per support
	
	n_c=[0,3,3,0]: Number of columns per support
	** abutment1, 2 there are no columns
		number of columns = 3 in each pie 1 and 2 
		
	q:	percent of the bridge weight
	
	k:  the increased parameter of  post-yield stiffness	  
	
	W_SS: Weight of superstructure including railings, curbs,and barriers to the permanent loads
	
	W_PP: Weight of piers participating with superstructure in dynamic response
	
	W_eff= W_SS + W_PP: Effective weight
	
	W=[W_1,W_2,...,W_m]: Weight of superstructure at each support
	
	K_sub=[K_sub_abut1, K_sub_pie1, K_sub_pie2, K_sub_abut2]: Stiffness of each support in both longitudinal and transverse directions of the bridge
	** For the abutments, take Ksub,j to be a large number, say 10,000 kips/in.
	
	angle_skew: Angle of skew
	
	PGA,S_1, S_S: Acceleration coefficients for bridge site are given in design statement
	
	SiteClass:  "A", "B", "C","D","E"
	
	epsilon: tolerance
	
	d: set initial guess for the first iteration
 
	"""
	#Isolator_Type = ['Lead-rubber bearing','Spherical friction bearing','EradiQuake bearing']

# Calculate Response Spectrum:
	# Create a array of time:
	shape=200
	t=np.linspace(0,T_max,shape)
	# Call the Response Spectrum function
	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 
	

	 # Plot the design response spectrum
	
	if plot_action==True:
		
		fig, ax = plt.subplots()
		ax.plot(t, C_sm)
		ax.set_title(f"Design Response Spectrum for PGA={PGA}, S_S={S_S}, S_1={S_1}, SiteClass={SiteClass}")
		ax.set_xlabel('Period')
		ax.set_ylabel('Acceleration')
		plt.show()



	##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# B2.1.1—Step A: Identifty Bridge Data
	## B2.1.1.2—Step A2: Seismic Hazard
	
	# B2.1.2—STEP B: ANALYZE BRIDGE FOR EARTHQUAKE LOADING IN LONGITUDINAL DIRECTION
	## B2.1.2.1—STEP B1: SIMPLIFIED METHOD
	### B2.1.2.1.1—Step B1.1: Initial System Displacement and Properties
		
	##%% Assume that the initial value of displacement d approximates 2.0
	#d=1.84
	i=1
	
	#d=10*S_D1
	#d=1.84
	data=dict()



	#print(f'iteration:{i} ')
	#print(f'd={d}')

	##%% Calculate characteristic strength, Q_d
	Q_d=0.05*W_SS

	##%% Calculate Post-yield stiffness, K_d
	K_d=0.05*(W_SS/d)

	### B2.1.2.1.2—Step B1.2: Initial Isolator Properties at Supports

	##%% Calculate the characteristic strength, Q_dj

	Q_dj=[Q_d*(W[j]/W_SS) for j in range(m)]

	##%% Calculate postelastic stiffness, K_dj

	K_dj= [K_d*(W[j]/W_SS) for j in range(m)]

	while True:
		### B2.1.2.1.3—Step B1.3: Effective Stiffness of Combined Pier and Isolator System

		##%% Calculate alpha_j

		alpha_j = [(K_dj[j]*d+Q_dj[j])/(K_sub[j]*d-Q_dj[j]) for j in range(m)]

		##%% Calculate the effective stiffness, K_effj

		K_effj=[(alpha_j[j]*K_sub[j])/(1+alpha_j[j]) for j in range(m)]

		### B2.1.2.1.4—Step B1.4: Total Effective Stiffness

		##%% Calculate the total effective stiffness, Keff, of the bridge:

		K_eff= sum(K_effj)
		#print(f'K_eff: {K_eff}')

		### B2.1.2.1.5—Step B1.5: Isolation System Displacement at Each Suppor

		##%% Calculate the displacement of the isolation system, d_isolj

		d_isolj=  [d/(1+ alpha_j[j]) for j in range(m)]
		
		#print(f'd_isolj: {d_isolj}')

		### B2.1.2.1.6—Step B1.6: Isolation System Stiffness at Each Support

		##%% Calculate the effective stiffness of the isolation system at support “j”, Kisol,j, for all supports

		K_isolj=[Q_dj[j]/d_isolj[j]+ K_dj[j] for j in range(m)]

		### 2.1.2.1.7—Step B1.7: Substructure Displacement at Each Support, d_subj

		d_subj= [d- d_isolj[j] for j in range(m)]

		### B2.1.2.1.8—Step B1.8: Lateral Load in Each Substructure Support

		##%% Calculate the shear at support “j”, Fsub,j, for all supports:

		F_subj= [K_sub[j]* d_subj[j] for j in range(m)]

		### B2.1.2.1.9—Step B1.9: Column Shear Force at Each Support

		F_coljk = [F_subj[j]/n_c[j] if n_c[j] != 0 else 0 for j in range(m)]

		### B2.1.2.1.10—Step B1.10: Effective Period and Damping Ratio

		##%% Calculate the effective period, T_eff 

		W_eff= W_SS + W_PP #  Effective weight, W_eff
		g=386.4 # (in./s^2) or 9.815(m/s^2)

		T_eff=2*np.pi* (W_eff/(g*K_eff))**(1/2)
		#print(f'T_eff: {T_eff}')

		##%% Calculate the viscous damping ratio, ξ , of the bridge

		d_yj=[0]*m # taking d_yj=0

		numerator=2*sum([Q_dj[j]*(d_isolj[j]-d_yj[j]) for j in range(m)])

		denominator=np.pi*sum([K_effj[j]*(d_isolj[j]+d_subj[j])**2 for j in range(m)])

		xi= numerator/denominator

		### B2.1.2.1.11—Step B1.11: Damping Factor

		##%% Calculate the damping factor, B_L

		B_L=Piecewise(((xi/0.05)**0.3,xi<0.3),(1.7, xi>=0.3))

		##%%  Calculate the displacement, d_new

		d_new=(9.79*S_D1*T_eff)/B_L
		
		#print(f'd_new: {d_new}')
	   #column_names_latex = 
		
		df = pd.DataFrame({"Pier": ["Abut1", "Pier1", "Pier2", "Abut2"],
							"d": d, "d_new": d_new,
							"$$Q_d$$": Q_d,
							"$$K_d$$": K_d,	  
							"$$Q_{d,j}$$":Q_dj,
							"$$K_{d,j}$$":K_dj,
							"$\\alpha_j$": alpha_j, 
							"$$K_{eff,j}$$": K_effj,
							"$$d_{isol,j}$$":d_isolj,
							"$$K_{isol,j}$$": K_isolj,
							"$$d_{sub,j}$$": d_subj, 
							"$$F_{sub,j}$$":F_subj ,
							"$$ F_{col,j,k}$$":F_coljk,
							"$$T_{eff}$$": T_eff,
							"$$K_{eff}$$":K_eff,
							"$$\\xi$$":xi,
							"$$B_{L}$$": B_L})



		# Convert all columns to float
		for column in df.columns[1:]:
			df[column] = df[column].astype(float)
		
		data[i]=df

		##%%%%%%%%%%%%%%%%%%%%%
		##%% Calculate the diference, abs(d_new-d) 

		difference=abs((d_new-d)/d)
		#delta=difference/d

		##%% Check the convergence condition:

		#if difference> epsilon:
		if difference> tol:
			d=d_new
			i+=1

		else:
			break
	print(f'The problem reaches convergence after  {i} iterations')
	
	# The minimum displacement requirement given by:
	
	d_min=(8*S_D1*T_eff)/B_L
	
	print(f'The minimum displacement requirement given by: d_min={d_min: .2f}')
	
	   
	for i in data.keys():
		data[i]['Iteration']=i # Add new column to track the iteration
		for col in data[i].columns:
			if col in ["$\\alpha_j$", "$$d_{sub,j}$$"]:
				data[i][col]=data[i][col].apply(scientific_format)
			else:
				data[i][col]=data[i][col]. apply(round_values, n=2)
		data[i].set_index(['Iteration'], inplace=True)
		#data[k].set_index(['Iteration',"d", "d_new","$$Q_d$$","$$K_d$$","$$T_{eff}$$","$$K_{eff}$$","$$\\xi$$", "$$B_{L}$$", 'Pier'], inplace=True)
	if latex_format==False:
		for i in data.keys():
			data[i]=data[i].rename(columns={
				"$$Q_d$$": "Q_d",
				"$$K_d$$": "K_d",
				"$$Q_{d,j}$$": "Q_dj",
				"$$K_{d,j}$$": "K_dj",
				"$\\alpha_j$": "alpha_j",
				"$$K_{eff,j}$$": "K_effj",
				"$$d_{isol,j}$$": "d_isolj",
				"$$K_{isol,j}$$": "K_isolj",
				"$$d_{sub,j}$$": "d_subj",
				"$$F_{sub,j}$$": "F_subj",
				"$$ F_{col,j,k}$$": "F_coljk",
				"$$T_{eff}$$": "T_eff",
				"$$K_{eff}$$": "K_eff",
				"$$\\xi$$": "xi",
				"$$B_{L}$$": "B_L"
				})
	 # Concatenate the DataFrames from each iteration
	concat_df=pd.DataFrame()
   
	for i, df in data.items():
		concat_df = pd.concat([concat_df, df], ignore_index=False)
	#return list(data.values())[-1]
	return data
###############################################################################
	
def Multimode(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,q,k,tol,T_max, Isolator_Type,latex_format=True):
	
	#isolator_type = ["friction-based isolators", 'others'] 
	
	"""
	m: Number of supports
	
	n: Number of girders per support
	
	n_c=[0,3,3,0]: Number of columns per support
	** abutment1, 2 there are no columns
		number of columns = 3 in each pie 1 and 2 
		
	q:	percent of the bridge weight
	
	k:  the increased parameter of  post-yield stiffness	  
	
	W_SS: Weight of superstructure including railings, curbs,and barriers to the permanent loads
	
	W_PP: Weight of piers participating with superstructure in dynamic response
	
	W_eff= W_SS + W_PP: Effective weight
	
	W=[W_1,W_2,...,W_m]: Weight of superstructure at each support
	
	K_sub=[K_sub_abut1, K_sub_pie1, K_sub_pie2, K_sub_abut2]: Stiffness of each support in both longitudinal and transverse directions of the bridge
	** For the abutments, take Ksub,j to be a large number, say 10,000 kips/in.
	
	angle_skew: Angle of skew
	
	PGA,S_1, S_S: Acceleration coefficients for bridge site are given in design statement
	
	SiteClass:  "A", "B", "C","D","E"
	
	epsilon: tolerance
	
	d: set initial guess for the first iteration
 
	"""

# Calculate Response Spectrum:
	# Create a array of time:
	shape=200
	t=np.linspace(0,T_max,shape)
	# Call the Response Spectrum function
	C_sm, F_pga, F_a, F_v, A_S, S_DS,S_D1=AASHTO(t, PGA,S_S,S_1,SiteClass) 
	


## Step 1: Iteration
	data=Iteration(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,q,k,tol,T_max, Isolator_Type,latex_format=False,plot_action=True, d=2.0)
	display(data)
	d=data.d.unique()[0]
	d_new=data.d_new.unique()[0]
	print(type(d))
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
	T_eff=data.T_eff.unique()[0]
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

	N=int((0.8*T_eff*shape)/T_max)
		
	C_sm[N:]=C_sm[N:]/B_L # C_sm is calculated by Response Spectrum, and B_L is the result from the last  convergence step.
	## Show the plot here
	plt.plot(t, C_sm)
	plt.title(f"Design Response Spectrum Multi Modes for PGA={PGA}, S_S={S_S}, S_1= {S_1}, SiteClass={SiteClass}")
	plt.xlabel(f'Period')
	plt.ylabel(f'Acceleration')
	plt.show()
	
	### Begin iteration
	iterration =1
	d=d

	while True:
		## B2.1.2.2.1—Step B2.1:Characteristic Strength
	
		##%% Calculate the characteristic strength, Qd,i, and postelastic stiffness, Kd,i, of each isolator “i” 
		
		Q_di=[Q_dj[j]/n for j in range(m)]
		
		
		K_di=[K_dj[j]/n for j in range(m)]
		
		#print(f'K_di={K_di}')
	
		## B2.1.2.2.2—Step B2.2: Initial Stiffness and Yield Displacement
	
		##%% Calculate the initial stiffness, Ku,i, and the yield displacement, dy,i, for each isolator “i”
	
		if Isolator_Type == "friction-based isolators":
			K_ui =[np.inf for j in range(m)] # n=6, is the numbers of isolator per each support 
			d_yi=[0]*m
		else:
			K_ui=[10*K_di[j] for j in range(m)]
			d_yi=[Q_di[j]/(K_ui[j]-K_di[j]) for j in range(m)]
		#print(f'K_ui={K_ui}')
		#print(f'd_yi={d_yi}')
	
		## B2.1.2.2.3—Step B2.3: Isolator Effective Stiffness, Kisol,i
	
		##%% Calculate the isolator stiffness, Kisol,i, of each isolator “i”
		
		
		k_isoli=[K_isolj[j]/n for j in range (m)] #(sample: n=6, m=4)
		
		#print(f'k_isoli={k_isoli}')
	
		## B2.1.2.2.4—Step B2.4: ThreeDimensional Bridge Model
	
		## B2.1.2.2.5—Step B2.5: Composite Design Response Spectrum
	   
	
		## B2.1.2.2.6—Step B2.6: Multimode Analysis of Finite Element Model
		
		#( Call the d_isolj from the previuos calculation)
	
	
	
	
	
		##%% Recalculate system damping ratio, ξ :
		
			
		# Update d_soli 
		df=Iteration(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,q,k,tol,T_max, Isolator_Type,latex_format=False,plot_action=False, d=d_new)
		d_isoli=df.d_isolj.to_list()
		print(f'd_isoli={type(d_isoli)}')
		#d1=df.d_new.unique()[0]
		#d_subj=df.d_subj.to_list()
		
		#print(d_subj)

		tol=abs((d_new-d)/d)
		
		if tol<=0.02:
			break
		else:
			print('continue iteration')
			
				
				
			## B2.1.2.2.8—Step B2.8: Update Kisol,i, Keff,j, ξ , and BL
		
			##%% Use the calculated displacements in each isolator element to obtain new values of Kisol,i for each isolator
		
			K_isoli=[Q_di[j]/d_isoli[j]+K_di[j] for j in range(m)] # (d_isoli in the document) is d_isolj in the convergence step
			
			print(f'K_isoli={K_isoli}')
			###%% Recalculate K_effj
		
			K_effj=[(K_sub[j]*K_isoli[j]*n)/(K_sub[j]+K_isoli[j]*n) for j in range(m)]
			
			print(f'K_effj={K_effj}')
		
		
	
			
		
		
		
	# =============================================================================
	# 	
	# 	##%% Calculate alpha_j
	# 
	# 	alpha_i = [(K_dj[i]*d_multimode+Q_dj[i])/(K_sub[i]*d_multimode-Q_dj[i]) for i in range(m)]
	# 
	# 	##%% Calculate the displacement of the isolation system, d_isolj
	# 
	# 	d_isoli=  [d_multimode/(1+ alpha_i[i]) for i in range(m)]
	# 
	# =============================================================================
			#print(f'd_isoli={d_isoli}')
			numerator=2*sum([n*Q_di[j]*(d_isoli[j]-d_yi[j]) for j in range(m)])
			denominator=np.pi*sum([K_effj[j]*(d_isoli[j]+d_subj[j])**2 for j in range(m)]) 
		
			xi=numerator/denominator
			#print(f'xi={xi}')
			
			## Recalculate system damping factor, BL:
			B_L=Piecewise(((xi/0.05)**0.3,xi<0.3),(1.7, xi>=0.3))
			#print(f'B_L={B_L}')
			#Recalculate Keff
		
			K_eff=sum(K_effj)
			
			#print(f'K_eff={K_eff}')
			
			#print(f'W_eff {W_eff}')
			
			#print(f'g={g}')
		
			# Recalculate T_eff
			W_eff= W_SS + W_PP #  Effective weight, W_eff
			g=386.4 # (in./s^2) or 9.815(m/s^2)
			T_eff= 2*np.pi*(W_eff/(g*K_eff))**(1/2)
			
			#print(f'T_eff={T_eff}')
		
			## Recalculate d
			d_multimode_new=(9.79*S_D1*T_eff)/B_L
			#print(f'd_multimode_new={d_multimode_new}')
	
			d=d_multimode_new
			iter_multimode+=1

	print(f'd_multimode_new {round(d_multimode_new,2)}')

	print(f'Numbers of iteratations to get convergence check: {iter_multimode}')
	
	
	#superstructure displacements in the longitudinal (xL) and transverse (yL) directions are:

	x_L=d_isoli[0]*np.cos(angle_skew)
	y_L=d_isoli[0]*np.sin(angle_skew)
	print(f'Superstructure displacements in the longitudinal:\n x_L={round(x_L,2)}\n y_L={round(y_L,2)}')

	# isolator displacements in the longitudinal (uL) and transverse (vL) directions are:
	u_L=[]
	v_L=[]
	# Abutment:
	u_L.append(d_isoli[0]*np.cos(angle_skew))
	v_L.append(d_isoli[0]*np.sin(angle_skew))
	print(f'isolator displacements in the longitudinal (uL) and transverse (vL) directions are:')
	print(f'Abutments: u_L={round(u_L[0],2)}, v_L={round(v_L[0],2)}')

	#Piers:

	u_L.append(d_isoli[1]*np.cos(angle_skew))
	v_L.append(d_isoli[1]*np.sin(angle_skew))
	print(f'Piers: u_L={round(u_L[1],2)}, v_L={round(v_L[1],2)}')

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
