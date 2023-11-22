# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
from Response_Spectrum import AASHTO
import matplotlib.pyplot as plt
import numpy as np

#%%

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
#params=[m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,T_max, Isolator_Type,q,k,tol]
#def B1(m,n,n_c,W_SS, W_PP,W,K_sub,angle_skew,PGA, S_1,S_S, SiteClass,T_max, Isolator_Type,q,k,tol,latex_format=True,plot_action=False):
def B1(params,latex_format=True,plot_action=False):
	m, n, n_c, W_SS, W_PP, W, K_sub, angle_skew, PGA, S_1, S_S, SiteClass, T_max, Isolator_Type, q, k, tol = params
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
	
	print(f'F_pga={F_pga}, F_a={F_a}, F_v={F_v}, S_D1={S_D1}')
	

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

	
	d0=10*S_D1

	data=dict()


	##%% Calculate characteristic strength, Q_d
	Q_d=q*W_SS
	#print(f'q={q}')
	#print(f'W_SS={W_SS}')
	#print(f'Q_d={Q_d}')
	##%% Calculate Post-yield stiffness, K_d
	K_d=k*(W_SS/d0)

	### B2.1.2.1.2—Step B1.2: Initial Isolator Properties at Supports

	##%% Calculate the characteristic strength, Q_dj

	Q_dj=[Q_d*(W[j]/W_SS) for j in range(m)]

	##%% Calculate postelastic stiffness, K_dj

	K_dj= [K_d*(W[j]/W_SS) for j in range(m)]

	d=d0
	i=1

	
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

		d_isolj=[d/(1+ alpha_j[j]) for j in range(m)]
		
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
		#numpy.piecewise(x, condlist, funclist, *args, **kw)[source]


		#B_L=np.piecewise(((xi/0.05)**0.3,xi<0.3),(1.7, xi>=0.3))
		B_L=np.piecewise(xi, [xi<0.3,xi>=0.3], [(xi/0.05)**0.3, 1.7])

		##%%  Calculate the displacement, d_new

		d_new=(9.79*S_D1*T_eff)/B_L
		d_isolj=[d_new/(1+ alpha_j[j]) for j in range(m)]
		
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
							"$$K_{isol,j}$$":K_isolj,
							"$$d_{sub,j}$$": d_subj, 
							"$$F_{sub,j}$$":F_subj,
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
				"$$K_{eff,j}$$":"K_effj",
				"$$d_{isol,j}$$":"d_isolj",
				"$$K_{isol,j}$$":"K_isolj",
				"$$d_{sub,j}$$":"d_subj",
				"$$F_{sub,j}$$":"F_subj",
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
##############################################################################


