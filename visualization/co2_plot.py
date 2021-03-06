import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import numpy as np
#---------- data reading

import matplotlib
params = {
   'axes.labelsize': 20,
   'font.size': 20,
   'legend.fontsize': 20,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18,
   'text.usetex': False,
   'figure.figsize': [14, 10],
   }
matplotlib.rcParams.update(params)


def iso_reading_co2(direct,core_names,sample_names):
    # read the all the sample iso xlsx data, store in dictionary
    iso = dict()
    for i,core_name in enumerate(core_names):
        iso['direct'] = direct
        iso[core_name] = dict()
        iso[core_name]['core_name'] = core_name
        #iso[core_name] ['core_name'] = core_name
        iso[core_name]['direct'] = direct + core_name + '/'
        for sample_name in sample_names[i]:
            iso[core_name][sample_name]=dict()
            iso[core_name][sample_name]['sample_name'] = sample_name

            # read isotherm data
            iso[core_name][sample_name]['iso_filename'] = iso[core_name][
                                                              'direct'] + 'ISO_' + core_name + '_' + sample_name + '_co2.csv'
            data_iso = pd.read_csv(iso[core_name][sample_name]['iso_filename'])
            iso[core_name][sample_name]['p_ads'] = data_iso['p_ads']
            iso[core_name][sample_name]['q_ads'] = data_iso['q_ads']

            # read psd data

            # get psd
            iso[core_name][sample_name]['psd_filename'] = iso[core_name][
                                                              'direct'] + 'PSD_' + core_name + '_' + sample_name + '_co2.csv'
            data_psd = pd.read_csv(iso[core_name][sample_name]['psd_filename'])
            iso[core_name][sample_name]['Davg_3flex'] = data_psd['Davg']

            iso[core_name][sample_name]['Vp_dlogD_3flex'] = data_psd['Vp_dlogD']
    return iso

def plot_color_index(sample_name):
    # input sample_name
    # output the color name for that sample
    flag = sample_name[0:2]
    colors={'HF':'r','In':'b','VF':'g'}
    color = colors[flag]
    #print(color)
    return color
#    return


def plot_isotherm_per_core_in_one_figure(iso,core_names_select,sample_names_select,save=False):


    direct = iso['direct']
    fig, axies = plt.subplots(4, 2)

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        if len(sample_names_select[i]) > 2:
            sample_VF = sample_names_select[i][2]
    # calculate the axis index
        ax_0 = i // 2
        ax_1 = i % 2
        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], 'bo-', markersize=10,
                               linewidth=2,label='Intact')
        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['p_ads'], iso[core_name][sample_F]['q_ads'], 'rD-', markersize=10,
                               linewidth=2,label='HFail')

        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso[core_name][sample_VF]['p_ads'], iso[core_name][sample_VF]['q_ads'], 'g^-',
                                   markersize=10,linewidth=2, label='VFail')

        if (ax_0 == 0) & (ax_1 == 0):
                axies[ax_0, ax_1].legend(loc=4)

        if i < 6:
            axies[ax_0, ax_1].set_xticks([])

            # plt.legend(handles=legends)
        if i < 3:
            axies[ax_0, ax_1].set_title('EF_'+core_name,fontsize=20, x=0.2, y=0.8)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + core_name, fontsize=20, x=0.2, y=0.8)

        axies[ax_0,ax_1].set_xlim(xmin=-0.001,xmax=0.023)
        axies[ax_0,ax_1].set_ylim(ymin=0)
        if i == 0:
            axies[ax_0,ax_1].set_yticks([0,0.1,0.2,0.3,0.4])
        elif i==5:
            axies[ax_0, ax_1].set_yticks([0,0.2, 0.2, 0.4, 0.6,0.8])
        xticks = np.arange(0, 0.03, 0.01)
        #axies[ax_0, ax_1].set_xticks(xticks)
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')
    fig.text(0.05, 0.6, 'Adsorption quantity (cm$^3$/g)', rotation=90)
            #axies[ax_0,ax_1].set_xlabel('a')
    fig.text(0.5, 0.01, 'Relative pressure (p/p$^0$)', ha='center')
    plt.tight_layout(pad=4,w_pad=2,h_pad=0.5)

    if save:
        plt.savefig(direct + 'failed_intact_isotherm_8_subplots.png')
    plt.show()


def plot_psd_per_core_in_one_figure(iso,core_names_select,sample_names_select,type ='log',save=False):
    direct = iso['direct']
    volume_type ='Vp_dlogD_3flex'

    fig, axies = plt.subplots(2, 4, figsize=(14, 10))

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        if len(sample_names_select[i]) > 2:
            sample_VF = sample_names_select[i][2]
    # calculate the axis index
        ax_0 = i // 4
        ax_1 = i % 4
        #iso[core_name][sample_name][] = psd.Davg
        #iso[core_name][sample_name]['Vp'] = psd.Vp
        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['Davg_3flex']/10, iso[core_name][sample_In][volume_type], 'b-', markersize=8,
                               linewidth=2,alpha=0.7,label='In')
        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['Davg_3flex']/10, iso[core_name][sample_F][volume_type], 'r-', markersize=8,
                               linewidth=2,alpha=0.7,label='F')

        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso[core_name][sample_VF]['Davg_3flex']/ 10, iso[core_name][sample_VF][volume_type],
                                   'g-', markersize=8,
                                   linewidth=2, alpha=0.7, label='VF')

        axies[ax_0,ax_1].legend(loc=2,fancybox=True, framealpha=0.5)

        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['Davg'], iso[core_name][sample_F][volume_type],s=50,facecolors='none',linewidths=1,edgecolors='k')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['Davg'], iso[core_name][sample_In][volume_type], s=50,facecolors='none', linewidths=1, edgecolors='k')

        axies[ax_0, ax_1].set_title(core_name)
        #axies[ax_0,ax_1].set_xscale('log')
        #axies[ax_0,ax_1].set_xlim([1,10**4])
        axies[ax_0,ax_1].set_ylim(ymin=0)

        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        if ax_1 == 0:
            if type == 'log':
                axies[ax_0,ax_1].set_ylabel('dv/dlog pore volume')
            else:
                axies[ax_0, ax_1].set_ylabel('pore volume')
        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')
        fig.text(0.5, 0.04, 'pore size (nm)', ha='center')
    #plt.tight_layout(pad=1,w_pad=0.1,h_pad=2)
    plt.tight_layout()
    plt.show()

def plot_psd_N2_CO2_per_core_in_one_figure(iso_N2,iso_CO2,core_names_select,sample_names_select,type ='log',save=False):
    direct = iso_N2['direct']
    volume_type ='Vp_dlogD_3flex'
    ylim_max = [0.016,0.018,0.020,0.022,0.016,0.018,0.014,0.016]
    fig, axies = plt.subplots(4, 2)

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        if len(sample_names_select[i]) > 2:
            sample_VF = sample_names_select[i][2]
    # calculate the axis index
        ax_0 = i // 2
        ax_1 = i % 2
        #iso[core_name][sample_name][] = psd.Davg
        #iso[core_name][sample_name]['Vp'] = psd.Vp

        axies[ax_0, ax_1].plot([1, 1], [0, 1], 'k-', linewidth=2, alpha=1)
        axies[ax_0, ax_1].plot([2.1, 2.1], [0, 1], 'k--', linewidth=2, alpha=0.5)

        axies[ax_0, ax_1].plot(iso_N2[core_name][sample_In]['Davg_3flex']/10, iso_N2[core_name][sample_In][volume_type], 'b-', markersize=8,
                               linewidth=2,label='Intact')
        axies[ax_0, ax_1].plot(iso_N2[core_name][sample_F]['Davg_3flex']/10, iso_N2[core_name][sample_F][volume_type], 'r-', markersize=8,
                               linewidth=2,label='HFail')




        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso_N2[core_name][sample_VF]['Davg_3flex']/ 10, iso_N2[core_name][sample_VF][volume_type],
                                   'g-', markersize=8,
                                   linewidth=2, label='VFail')
        if i==0:
            axies[ax_0, ax_1].legend(loc=2, fancybox=True, framealpha=0.9,prop={'size':17})



        axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_In]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_In][volume_type], 'b-', markersize=6,
                               linewidth=2, )
        axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_F]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_F][volume_type], 'r-', markersize=8,
                               linewidth=2)
        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_VF]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_VF][volume_type],
                               'g-', markersize=8,linewidth=2,  label='VFail')



        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['Davg'], iso[core_name][sample_F][volume_type],s=50,facecolors='none',linewidths=1,edgecolors='k')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['Davg'], iso[core_name][sample_In][volume_type], s=50,facecolors='none', linewidths=1, edgecolors='k')
        axies[ax_0,ax_1].set_yticks([0,0.005,0.01,0.015,0.02])

        #axies[ax_0,ax_1].set_xscale('log')
        axies[ax_0,ax_1].set_xlim([0.3,250])
        axies[ax_0,ax_1].set_ylim([0,ylim_max[i]])
        axies[ax_0,ax_1].set_xscale('log')
        axies[ax_0, ax_1].set_xticks([0.3,1, 10, 100])
        axies[ax_0, ax_1].set_xticklabels(['0.3','1', '10', '100'])

        if i < 6:
            axies[ax_0, ax_1].set_xticks([])

            # plt.legend(handles=legends)
        if i < 3:
            axies[ax_0, ax_1].set_title('EF_'+core_name,fontsize=25, x=0.7, y=0.05)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + core_name, fontsize=25, x=0.7, y=0.05)
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')

    fig.text(0.5, 0.04, 'Pore width (nm)', ha='center')
    fig.text(0.05, 0.6, 'dV/dlog(w) (cm$^3$/g)', rotation=90)
            #axies[ax_0,ax_1].set_xlabel('a')
    #fig.text(0.5, 0.01, 'Relative pressure (p/p$^0$)', ha='center')
    plt.tight_layout(pad=4,w_pad=1,h_pad=0.1)
    plt.show()


#--------
def plot_psd_diff_N2_CO2_per_core_in_one_figure(iso_N2,iso_CO2,core_names_select,sample_names_select,type ='log',save=False):
    direct = iso_N2['direct']
    volume_type ='Vp_dlogD_3flex'
    ylim_max = [0.016,0.018,0.020,0.022,0.016,0.018,0.014,0.016]
    fig, axies = plt.subplots(4, 2, figsize=(12, 12))

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        if len(sample_names_select[i]) > 2:
            sample_VF = sample_names_select[i][2]
    # calculate the axis index
        ax_0 = i // 2
        ax_1 = i % 2
        #iso[core_name][sample_name][] = psd.Davg
        #iso[core_name][sample_name]['Vp'] = psd.Vp

        axies[ax_0, ax_1].plot([1, 1], [0, 1], 'k-', linewidth=2, alpha=1)
        axies[ax_0, ax_1].plot([2.1, 2.1], [0, 1], 'k--', linewidth=2, alpha=0.5)


        axies[ax_0, ax_1].plot(iso_N2[core_name][sample_In]['Davg_3flex']/10, iso_N2[core_name][sample_In][volume_type], 'b-', markersize=8,
                               linewidth=2,label='Intact')
        axies[ax_0, ax_1].plot(iso_N2[core_name][sample_F]['Davg_3flex']/10, iso_N2[core_name][sample_F][volume_type], 'r-', markersize=8,
                               linewidth=2,label='HFail')




        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso_N2[core_name][sample_VF]['Davg_3flex']/ 10, iso_N2[core_name][sample_VF][volume_type],
                                   'g-', markersize=8,
                                   linewidth=2, label='VFail')
        if i==0:
            axies[ax_0, ax_1].legend(loc=2, fancybox=True, framealpha=0.9,prop={'size':17})



        axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_In]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_In][volume_type], 'b-', markersize=6,
                               linewidth=2, )
        axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_F]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_F][volume_type], 'r-', markersize=8,
                               linewidth=2)
        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso_CO2[core_name][sample_VF]['Davg_3flex'] / 10,
                               iso_CO2[core_name][sample_VF][volume_type],
                               'g-', markersize=8,linewidth=2,  label='VFail')



        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['Davg'], iso[core_name][sample_F][volume_type],s=50,facecolors='none',linewidths=1,edgecolors='k')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['Davg'], iso[core_name][sample_In][volume_type], s=50,facecolors='none', linewidths=1, edgecolors='k')
        axies[ax_0,ax_1].set_yticks([0,0.005,0.01,0.015,0.02])

        #axies[ax_0,ax_1].set_xscale('log')
        axies[ax_0,ax_1].set_xlim([0.3,250])
        axies[ax_0,ax_1].set_ylim([0,ylim_max[i]])
        axies[ax_0,ax_1].set_xscale('log')
        axies[ax_0, ax_1].set_xticks([0.3,1, 10, 100])
        axies[ax_0, ax_1].set_xticklabels(['0.3','1', '10', '100'])

        if i < 6:
            axies[ax_0, ax_1].set_xticks([])

            # plt.legend(handles=legends)
        if i < 3:
            axies[ax_0, ax_1].set_title('EF_'+core_name,fontsize=15, x=0.7, y=0.05)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + core_name, fontsize=15, x=0.7, y=0.05)
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')

    fig.text(0.5, 0.04, 'Pore width (nm)', ha='center')
    fig.text(0.05, 0.6, 'dV/dlog(w) (cm$^3$/g)', rotation=90)
            #axies[ax_0,ax_1].set_xlabel('a')
    #fig.text(0.5, 0.01, 'Relative pressure (p/p$^0$)', ha='center')
    plt.tight_layout(pad=4,w_pad=1,h_pad=0.1)
    plt.show()
