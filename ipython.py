import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

from visualization import N2_plot


#------------- main function


direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_trixial/'

core_names = [
    '1_223',
    '2_50',

    '2_93',
    '3_14',

    '3_42',
    '3_53',

    '4_14',
    '4_34']
sample_names = [
    ['HF_1','HF_2','In_1','In_2','VF_1','VF_2'],
    ['HF_1','HF_2','In_1','In_2'],

    ['HF_1','HF_2','In_1','In_2'],
    ['HF_1', 'HF_2', 'HF_3','HF_4','In_1','In_2','In_3'],

    ['HF_1','HF_2','In_1','In_2'],
    ['HF_1', 'HF_2', 'HF_3','In_1','In_2','In_3'],

    ['HF_1', 'HF_2', 'HF_3','In_1','In_2','In_3'],
    ['HF_1', 'HF_2', 'HF_3','In_1','In_2','In_3'],

]

core_names_select = [
    '1_223',
    '1_223',
    '2_50',

    '2_93',
    '3_14',

    '3_42',
    '3_53',

    '4_14',
    '4_34'
]

sample_names_select = [
    ['HF_1', 'In_1'],
    ['VF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_2', 'In_3']
]


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
    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_2', 'In_3']
]



BET_select_1 = np.array([
    [4.8029, 4.3175],
    [6.1912, 6.2181 ],

    [7.3412,6.8987],
    [11.2472, 8.0645],

    [10.4431,12.0849],
    [7.3647,6.7764],

    [8.7654, 6.9129],
    [7.1313, 7.2602],

])


iso = N2_plot.iso_reading(direct,core_names,sample_names)
#N2_plot.plot_porosity_bar(iso,core_names,sample_names,save = True)
#N2_plot.plot_isotherm_per_core(iso,core_names,sample_names,save=True)
#N2_plot.plot_porosity_bar(iso,core_names,sample_names,save =True)
#N2_plot.plot_porosity_ratio_all_sample(iso,core_names_select,sample_names_select,True)
#N2_plot.plot_isotherm_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
#N2_plot.plot_intact_isotherm_all_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
#N2_plot.plot_psd_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,'log',True)

#print(BET_select_1[1,1])
#plot_BET_surface_bar(iso,core_names_select_1,BET_select_1,True)
#plot_total_porosity_bar(iso,core_names_select_1,sample_names_select_1)
'''
for i, core_name in enumerate(core_names_select):
    Fail_micro = list()
    Intact_meso = list()
    xticklabel = list()
    sample_F = sample_names_select[i][0]
    sample_In = sample_names_select[i][1]
    Fail_micro.append(iso[core_name][sample_F]['vpore_micro'])
    Intact_meso.append(iso[core_name][sample_In]['vpore_meso'])

    xticklabel.append(sample_names)
    data_micro = tuple(data_micro)
    data_meso = tuple(data_meso)
    # xticklabel = tuple(xticklabel)

    width = 0.35
    ind = np.arange(len(data_micro))
    print(data_micro, data_meso)
    figure = plt.figure(core_name)
    # fig,ax = plt.subplots()
    rects1 = plt.bar(ind, data_micro, width, color='r')
    rects2 = plt.bar(ind, data_meso, width, bottom=data_micro, color='y')
    # ax.set_xticks()
    plt.title('pore_' + core_name)
    plt.xticks(ind + width / 2, xticklabel)
    plt.legend([rects1[0], rects2[0]], ['micro', 'meso'], fancybox=True, framealpha=0.5)
    # plt.legend()
    if save:
        plt.savefig(direct + 'Pore_' + core_name + '.png')



#psd.plot_isotherm()
#psd.plot_BJH_psd()
#plt.show()
#N = 5
#menMeans = (20, 35, 30, 35, 27)
#womenMeans = (25, 32, 34, 20, 25)
#menStd = (2, 3, 4, 1, 2)
#womenStd = (3, 5, 2, 3, 3)
#ind = np.arange(N)    # the x locations for the groups
#width = 0.35       # the width of the bars: can also be len(x) sequence

#figure = plt.figure()
#ind = np.arange(2)
#data = (psd.vpore_micro,psd.vpore_meso)
#p1 = plt.bar(ind,data)
#plt.xticks(ind, ())
#plt.show()

#-------- plotting data -----------------------------
'''

