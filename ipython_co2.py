import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
import pandas as pd
from visualization import co2_plot


#------------- main function


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

#data_iso = pd.read_csv(direct+'1_223/'+'ISO_1_223_HF_1_co2.csv')
iso = co2_plot.iso_reading_co2(direct,core_names_select_1,sample_names_select_1)
#N2_plot.plot_porosity_bar(iso,core_names_select_1,sample_names_select_1,save = True)
#N2_plot.plot_isotherm_per_core(iso,core_names_select_1,sample_names_select_1,save=True)
#N2_plot.plot_porosity_bar(iso,core_names,sample_names,save =True)
#N2_plot.plot_porosity_ratio_all_sample(iso,core_names_select,sample_names_select,True)
#co2_plot.plot_isotherm_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
#N2_plot.plot_intact_isotherm_all_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
co2_plot.plot_psd_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,'log',True)
#co2_plot.plot_color_index(sample_names_select_1[0][2])
#print(BET_select_1[1,1])
#plot_BET_surface_bar(iso,core_names_select_1,BET_select_1,True)
#plot_total_porosity_bar(iso,core_names_select_1,sample_names_select_1)




