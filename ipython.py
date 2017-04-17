import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('ggplot')
import pandas as pd
from visualization import N2_plot
from sklearn import linear_model

#------------- main function


direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/'

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
    ['HF_1', 'In_1','VF_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1'],

    ['HF_1', 'In_1'],
    ['HF_1', 'In_1','HF_2'],

    ['HF_1', 'In_1'],
    ['HF_2', 'In_3']
]


'''
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

CO2_porosity = np.array([
    [0.000719, 0.000651943 ],
    [0.000751227, 0.000948556],

    [0.000861417, 0.000806838],
    [0.001930066, 0.001541005],

    [0.001480599,0.00155312],
    [0.001555536,0.001403639],

    [0.001677593, 0.001144392],

    [0.001876338, 0.001964157]
])

ratio_co2_porosity = CO2_porosity[:,0]/CO2_porosity[:,1]
print('co2',ratio_co2_porosity)
N2_psd_total = np.array([
    [0.015337514,0.013503777],
    [0.018945354, 0.01894007],

    [0.022989719, 0.020594033],
    [0.020888031,0.014062324],

    [0.010213078, 0.012274276],
    [0.016805195,0.01496171],

    [0.008990891,0.008666781],
    [0.012957134,0.01512098],
])
N2_psd_micro = np.array([
    [0.000286071, 0.000232928],
    [0.000347841, 0.000365849],

    [0.00049224,0.000403205 ],
    [0.000851667,0.000672105],

    [0.00180981,0.00208449],
    [0.000910934,0.000869904],

    [0.00138376,0.000793318],
    [0.001422318,0.001565737],
])

kerogen_psd = np.array([0.368443827,0.325187733,  0.274936231,0.104999386,  0.153874654,0.162972189,
                        0.196988362, 0.120564368])


N2_psd_meso = N2_psd_total - N2_psd_micro

ratio_meso = N2_psd_meso[:,0] / N2_psd_meso[:,1]
ratio_micro = N2_psd_micro[:,0] / N2_psd_micro[:,1]
ratio_total = N2_psd_total[:,0] / N2_psd_total[:,1]
print(ratio_micro,ratio_meso)

ratio_BET = BET_select_1[:,0] / BET_select_1[:,1]
'''


#direct_co2 = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_CO2_trixial/'

iso = N2_plot.iso_reading_N2(direct,core_names,sample_names)
#N2_plot.plot_fractal_dimension_method(iso)
f1 = iso['3_42']['In_2']['fractal_1']
f2 = iso['3_42']['In_2']['fractal_2']
#N2_plot.save_fractal_number(iso,core_names_select_1,sample_names_select_1,True)
#f1,f2 = N2_plot.get_fractal_number(p,q,True)
#print(f1)
#print(f2)

print('yes')

#print(df_compare)
#ratio_mesos_BJH,ratio_micros_BJH,ratio_totals_BJH= N2_plot.get_porosity_ratio_all_sample_BJH(iso,core_names_select_1,sample_names_select_1)
file_prechar ='C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/precharacterization_data.xlsx'
df_xrd = pd.read_excel(file_prechar,sheetname='xrd')
df_toc = pd.read_excel(file_prechar,sheetname='toc')
df_kerogen = pd.read_excel(file_prechar,sheetname='kerogen_porosity')
file_post = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/post_data.xlsx'
df_BET =  pd.read_excel(file_post,sheetname='BET')
#df_BET['diff'] = df_BET['F']- df_BET['In']
#N2_plot.plot_mineral_bar(df_xrd)
df_N2_total =  pd.read_excel(file_post,sheetname='N2_total')
df_N2_meso =  pd.read_excel(file_post,sheetname='N2_meso')
df_micro =  pd.read_excel(file_post,sheetname='micro')
df_N2_micro =  pd.read_excel(file_post,sheetname='N2_micro')
df_fractal = pd.read_excel(file_post,sheetname='fractal')
df_BET_C = pd.read_excel(file_post,sheetname='BET_C')
#N2_plot.plot_BET_constant(df_BET_C,'In')
#df_N2_meso['diff'] = df_N2_meso['F']- df_N2_meso['In']
#df_N2_micro['diff'] = df_N2_micro['F']- df_N2_micro['In']

#df_CO2 =  pd.read_excel(file_post,sheetname='CO2')
#print(ratio_mesos)
#print(df_N2_meso['ratio'])

#N2_plot.plot_full_CO2_isotherm()



print(df_fractal)
id0 = 0
id1 = 10
df_compare = pd.DataFrame(data={
                                '1r_micro':np.array(df_micro['ratio'])[id0:id1],
                                '1r_N2_meso':np.array(df_N2_meso['ratio'])[id0:id1],
                                '1r_N2_total': np.array(df_N2_total['ratio'])[id0:id1],
                                #'N2_ini_meso': N2_psd_total[:,1] - N2_psd_micro[:,1],
                                #'N2_ini_micro': N2_psd_micro[:,1],
                                'fractal_In_D12': np.array(df_fractal['In_D12'])[id0:id1],
                                'fractal_In_D22': np.array(df_fractal['In_D22'])[id0:id1],
                                'fractal_diff_D12': np.array(df_fractal['diff_D12'])[id0:id1],
                                'fractal_diff_D22': np.array(df_fractal['diff_D22'])[id0:id1],
                                'N2_meso_diff': np.array(df_N2_meso['diff'])[id0:id1],
                                'micro_diff': np.array(df_micro['diff'])[id0:id1],
                                'BET_diff': np.array(df_BET['diff'])[id0:id1],
                                #'N2_In_micro': np.array(df_N2_micro['In'])[id0:id1],
                                #'N2_In_meso': np.array(df_N2_meso['In'])[id0:id1],
                                #'1r_CO2':np.array(df_CO2['ratio'])[id0:id1],
                                '1r_BET': np.array(df_BET['ratio'])[id0:id1],
                                'kerogen_meso': np.array(df_kerogen['N2_meso'])[id0:id1],
                                'kerogen_micro': np.array(df_kerogen['N2_micro'])[id0:id1],
                                'kerogen_total': np.array(df_kerogen['N2_total'])[id0:id1],
                                'quartz': np.array(df_xrd['quartz']),
                                #'HI_Tmax': np.array(df_toc['HI'][id0:id1])*np.array(df_toc['Tmax'][id0:id1]),
                                'HI': np.array(df_toc['HI']),
                                'Tmax': np.array(df_toc['Tmax']),
                                'clay': np.array(df_xrd['clay_total']) ,
                                'clay+toc': np.array(df_xrd['clay_total'])[id0:id1] +np.array(df_toc['TOC(%)'][id0:id1]) ,
                                'toc': np.array(df_toc['TOC(%)'][id0:id1])


                                },
                          index= df_N2_micro['core_name']
                       )
#print(df_compare)

#print(df_compare[3:])
colors = ['g','g','g',
     'b','b','b','b','b']
#scatter_matrix(df_compare[:], alpha=1, figsize=(12, 12),c=colors,s=100)

#scatter_matrix(df_compare[3:], alpha=0.5, figsize=(6, 6), diagonal='kde',s=300)
#plt.show()


#N2_plot.plot_df(df_compare,'fractal_In_D12','fractal_diff_D12',linear = False,xlim=[2.3,2.7,0],ylim=[-0.03,0.06,None])
#--------- checking list ------------------
#N2_plot.plot_df(df_compare,'1r_N2_meso','1r_micro',xlim=[0.6,1.8,1.0],ylim=[0.6,1.8,1.0],linear = False)
#N2_plot.plot_df(df_compare,'1r_BET','1r_micro',xlim=[0.8,1.6,1.0],ylim=[0.8,1.6,1.0],linear = True,x_inter=[0.9,1.3])
#N2_plot.plot_df(df_compare,'1r_BET','1r_N2_meso',xlim=[0.8,1.6,1.0],ylim=[0.8,1.6,1.0],linear = True,x_inter=[0.9,1.3])
#N2_plot.plot_df(df_compare,'BET_diff','micro_diff',xlim=[-2,4,0],ylim=[-0.0005,0.0012,0],linear = True,x_inter=[-1,2.5])
#N2_plot.plot_df(df_compare,'BET_diff','N2_meso_diff',xlim=[-2,4,0],ylim=[-0.0005,0.0012,0],linear = True,x_inter=[-1,2.5])
#N2_plot.plot_df(df_compare,'fractal_diff_D12','fractal_diff_D22',xlim=[-0.02,0.06,0],ylim=[-0.02,0.04,0],linear = False,x_inter=[-1,2.5])
#N2_plot.plot_df(df_compare,'fractal_diff_D12','1r_micro',xlim=[-0.02,0.06,None],ylim=[-0.02,0.04,None],linear = False,x_inter=[-0.01,0.015])
N2_plot.plot_df(df_compare,'1r_N2_meso','toc',xlim=[-0.02,0.06,None],ylim=[-0.02,0.04,None],linear = False,x_inter=[-0.01,0.015])
#plt.grid
plt.show()

#N2_plot.plot_df(df_compare,'kerogen_total','fractal_diff_D22',xlim=[2.3,2.7,None],ylim=[-0.03,0.06,None],linear = False)
#N2_plot.plot_df(df_compare,'fractal_diff_D22','fractal_diff_D12',plotline=True,linear = False,xlim=[-0.02,0.03],ylim=[-0.03,0.06],base=0)


#plt.show()
#N2_plot.plot_df(df_compare,'1r_N2_meso','1r_BET',True,True)
#plt.show()
#N2_plot.plot_df(df_compare,'1r_CO2','1r_BET',True,True)
#plt.show()
#N2_plot.plot_df(df_compare,'1r_BET','1r_N2_meso',True)
#N2_plot.plot_df(df_compare,'kerogen_micro','1r_N2_',False)
#N2_plot.plot_df(df_compare,'BET_diff','N2_diff_meso',False)
#N2_plot.plot_df(df_compare,'BET_diff','N2_diff_micro',False)
#N2_plot.plot_df(df_compare,'clay+toc','1r_N2_total',False)
#plt.show()
#print(df_compare)
#df_compare

#N2_plot.plot_porosity_bar(iso,core_names_select_1,sample_names_select_1,save = True)
#N2_plot.plot_isotherm_per_core(iso,core_names_select_1,sample_names_select_1,save=True)
#N2_plot.plot_porosity_bar(iso,core_names,sample_names,save =True)
#N2_plot.plot_porosity_ratio_all_sample(iso,core_names_select,sample_names_select,True)
#N2_plot.plot_isotherm_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,False)
#N2_plot.plot_intact_isotherm_all_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,True)
#N2_plot.plot_psd_per_core_in_one_figure(iso,core_names_select_1,sample_names_select_1,'my',True)

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

