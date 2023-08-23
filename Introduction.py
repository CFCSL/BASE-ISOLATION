#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:55:21 2023

@author: namnguyen
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import base64
from io import BytesIO
import Response_Spectrum as RS

st.title("Response Spectrum  AASHTO and Eurocode 8")


st.subheader("I. AASHTO ")

st.markdown("""

The response spectrum shall be calculated using the peak ground acceleration coefficient and the spectral acceleration coefficients scaled by the zero-, short- and long-period site factors, $F_{pga}$, $F_a$ and $F_v$ respectively.

""")

st.image("figures/Fig_1.jpeg")

st.markdown("""
			Site Factors $F_{pga}$, $F_a$ and $F_v$ specified in Tables 3.10.3.2-1, 3.10.3.2-2, and 3.10.3.2-3 shall be used in the zero-period, short-period range, and long-period range, respectively. These factors shall be determined using the Site Class given in Table 3.10.3.1-1 and the mapped values of the coefficients PGA, SS, and S1.
			""")
st.image("figures/PGA.jpeg")
st.image("figures/SS.jpeg")
st.image("figures/S1.jpeg")





st.subheader("II. Eurocode 8 ")

st.markdown("""
Horizontal elastic response spectra are calculated and plotted for two recommended types (Type 1 and Type 2) and for ground types A to E (5% damping). However, for vertical seismic (EN 1998-1:2004 (E)), all 5 ground types A, B, C, D and E have the same vertical spectrum.

""")
st.image("figures/Fig_2.jpeg")


st.subheader("Horizontal elastic response spectrum")
st.image("figures/Type_1.jpeg")
st.image("figures/Type_2.jpeg")

st.subheader("Vertical elastic response spectrum")
st.image("figures/Vertical.jpeg")





st.markdown("""for references see: \n
-  AASHTO LRFD BRIDGE DESIGN SPECIFICATIONS. Seventh edition, 2014, U.S. Customary Units, pp.3.91–3.93.
-  Eurocode 8: Design of structures for earthquake resistance EN 1998-1:2004(E) pp. 36-42.
""")



st.markdown("""
---
- The program developed by: 		Pedram Manouchehri & Nam Nguyen 
- User interface developed by:	Nam Nguyen 
- Independently Checked by:		Fernando Ávila
""")
