import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('ggplot')
import pandas as pd
from visualization import co2_plot
from visualization import N2_plot
import matplotlib
#------------- main function
# this is also for the paper

direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_CO2_triaxial/'

core_names_select_1 = [
    '1_223',
    '2_50',

    '2_93',
    '3_14',

    '3_42',
    '3_53',

    '4_14',
    '4_34'
]

sample_names_select_1 = [
    ['HF_1', 'In_1','VF_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1','HF_2'],

    ['HF_1', 'In_1'],
    ['HF_2', 'In_3']
]
direct_N2 = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/'
iso_N2 = N2_plot.iso_reading_N2(direct_N2,core_names_select_1,sample_names_select_1)
iso = co2_plot.iso_reading_co2(direct,core_names_select_1,sample_names_select_1)

#data_iso = pd.read_csv(direct+'1_223/'+'ISO_1_223_HF_1_co2.csv')
#iso = co2_plot.iso_reading_co2(direct,core_names_select_1,sample_names_select_1)
#N2_plot.plot_porosity_bar(iso,core_names_select_1,sample_names_select_1,save = True)
#N2_plot.plot_isotherm_per_core(iso,core_names_select_1,sample_names_select_1,save=True)
#N2_plot.plot_porosity_bar(iso,core_names,sample_names,save =True)
#N2_plot.plot_porosity_ratio_all_sample(iso,core_names_select,sample_names_select,True)

#N2_plot.plot_intact_isotherm_all_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
#co2_plot.plot_psd_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,'log',True)
#co2_plot.plot_color_index(sample_names_select_1[0][2])
#print(BET_select_1[1,1])
#plot_BET_surface_bar(iso,core_names_select_1,BET_select_1,True)
#plot_total_porosity_bar(iso,core_names_select_1,sample_names_select_1)

## plot----------------------
#---------------------------------------
params = {
   'axes.labelsize': 27,
   'font.size': 25,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [14, 18],
   }
matplotlib.rcParams.update(params)
co2_plot.plot_isotherm_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)

params = {
   'axes.labelsize': 27,
   'font.size': 25,
   'legend.fontsize': 25,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [14, 18],
   }
matplotlib.rcParams.update(params)

co2_plot.plot_psd_N2_CO2_per_core_in_one_figure(iso_N2,iso,core_names_select_1,sample_names_select_1,False)


