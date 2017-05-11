import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from visualization import N2_plot
filename = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/kerogen_N2.xlsx'
df_kerogen = pd.read_excel(filename,'ke_1_223')
#_,fracal2 = N2_plot.get_fractal_number(p,q,do_plot=False)