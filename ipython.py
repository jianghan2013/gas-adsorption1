import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import os
plt.style.use('ggplot')
from BJH_function import BJH_calculation
from BJH_function import read_3flex

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

iso = dict()
for i,core_name in enumerate(core_names):
    iso[core_name] = dict()
    iso[core_name]['core_name'] = core_name
    #iso[core_name] ['core_name'] = core_name
    iso[core_name]['direct'] = direct + core_name+'/'
    for sample_name in sample_names[i]:
        iso[core_name][sample_name]=dict()
        iso[core_name][sample_name]['sample_name'] = sample_name
        iso[core_name][sample_name]['filename'] = iso[core_name]['direct']+'ISO_'+core_name+'_'+sample_name +'_N2.csv'
        data = pd.read_csv(iso[core_name][sample_name]['filename'])
        iso[core_name][sample_name]['p_ads'] = data['p_ads']
        iso[core_name][sample_name]['q_ads'] = data['q_ads']


p_ads = iso['3_53']['In_2']['p_ads']
q_ads = iso['3_53']['In_2']['q_ads']

psd = BJH_calculation.BJH_method(p_ads,q_ads,use_pressure=False)
psd.do_BJH()
#psd.plot_isotherm()
#psd.plot_BJH_psd()
#plt.show()
#---------------------------------------------
'''
for i in range(0,len(core_names)):
    core_name = core_names[i]
    figure = plt.figure('ISO_'+core_name)
    legends = []
    for sample_name in sample_names[i]:
        if sample_name[0] == 'I':
            legend, = plt.plot(iso[core_name][sample_name]['p_ads'],iso[core_name][sample_name]['q_ads'],'*-',markersize=8,
                           label=iso[core_name]['core_name']+'_'+iso[core_name][sample_name]['sample_name'])
        else:
            legend, = plt.plot(iso[core_name][sample_name]['p_ads'],iso[core_name][sample_name]['q_ads'],'o-',markersize=8,
                           label=iso[core_name]['core_name']+'_'+iso[core_name][sample_name]['sample_name'])
        legends.append(legend)
    plt.title('ISO_'+ core_name)
    plt.legend(handles=legends,loc=2)
    plt.xlabel('relative pressure')
    plt.ylabel('adsorption quantity (cm3/g STP)')
    #plt.show()
    plt.savefig(direct+'ISO_'+core_name+'.png')
#for i, inputfile in enumerate(input_file_list):

    #data = pd.read_csv(direct+inputfile)
    #data = data.to_dict()
    #print(data)
'''
