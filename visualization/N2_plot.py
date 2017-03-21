from BJH_function import BJH_calculation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#---------- data reading
def iso_reading(direct,core_names,sample_names):
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
            iso[core_name][sample_name]['filename'] = iso[core_name]['direct']+ 'ISO_'+core_name+'_'+sample_name +'_N2.csv'
            data = pd.read_csv(iso[core_name][sample_name]['filename'])
            iso[core_name][sample_name]['p_ads'] = data['p_ads']
            iso[core_name][sample_name]['q_ads'] = data['q_ads']

            p_ads = iso[core_name][sample_name]['p_ads']
            q_ads = iso[core_name][sample_name]['q_ads']
            psd = BJH_calculation.BJH_method(p_ads, q_ads, use_pressure=True)
            psd.do_BJH()
            # get porosity
            iso[core_name][sample_name]['vpore_micro'] = psd.vpore_micro
            iso[core_name][sample_name]['vpore_meso'] = psd.vpore_meso
            iso[core_name][sample_name]['vpore_total'] = psd.vpore_total
            # get psd
            iso[core_name][sample_name]['Davg'] = psd.Davg
            iso[core_name][sample_name]['Vp'] = psd.Vp
            iso[core_name][sample_name]['Vp_dlogD'] = psd.Vp_dlogD
    return iso


#----------------------- visualization

#----------- bar chart
def plot_porosity_bar(iso,core_names,sample_names,save =False):
    '''
    inside one figure
    plot bar chart for all the sample in each core in terms of mesopore and micropore

    :param iso:
    :param core_names:
    :param sample_names:
    :param save:
    :return:
    '''

    fig,axies = plt.subplots(2,4,figsize=(14,10))

    for i,core_name in enumerate(core_names):
        data_micro = list()
        data_meso = list()
        xticklabel = list()
        for sample_name in sample_names[i]:
            data_micro.append(iso[core_name][sample_name]['vpore_micro'])
            data_meso.append(iso[core_name][sample_name]['vpore_meso'])
            xticklabel.append(sample_name)
        data_micro = tuple(data_micro)
        data_meso = tuple(data_meso)
        #xticklabel = tuple(xticklabel)

        # calculate the axis index
        ax_0 = i // 4
        ax_1 = i % 4
        print(i,ax_0,ax_1)

        width =0.35
        ind = np.arange(len(data_micro))
        #print(data_micro,data_meso)

        rects1 = axies[ax_0,ax_1].bar(ind,data_micro,width,color='r')
        rects2 = axies[ax_0,ax_1].bar(ind,data_meso,width,bottom=data_micro,color='y')
        #axies[ax_0,ax_1].set_ylim([0, 0.04])
        axies[ax_0,ax_1].set_title(core_name)

        #plt.title('pore_'+core_name)
        #plt.xticks(ind+width/2,xticklabel)
        plt.setp(axies[ax_0,ax_1],xticks=ind+width/2,xticklabels=xticklabel)
        #axies[ax_0,ax_1].set_xticks(xticklabel)
        #axies.legend([rects1[0],rects2[0]],['micro','meso'],fancybox=True, framealpha=0.5)
        #plt.legend()
        #figure = plt.figure(core_name)
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axies[0, :]], visible=False)
    #for k in range(1,4):
        #plt.setp([a.get_yticklabels() for a in axies[:, k]], visible=False)
    plt.tight_layout()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def plot_BET_surface_bar(iso,core_names,BET_select,save =False):
    '''

    '''
    direct = iso['direct']
#    figure = plt.figure()
    xticklabel = core_names
    fig,ax = plt.subplots(1,1)
    ind = np.arange(8)
    width =0.35
    #BET_select
    rects1 = ax.bar(ind, BET_select[:, 1], width, color='b')
    rects2 = ax.bar(ind+width+0.02,BET_select[:,0], width, color='r')
    #ax.legend()
    ax.set_xlim([-0.2,8])
    plt.setp(ax, xticks=ind + width+0.02, xticklabels=xticklabel)
    ax.legend([rects2[0], rects1[0]], ['F', 'In'], fancybox=True, framealpha=0.5)
    plt.xlabel('core names')
    plt.ylabel('specific surface area m2/g')

    if save:
        plt.savefig(direct + 'BET_all_core_bar.png')
    plt.show()

def plot_total_porosity_bar(iso,core_names_select,sample_names_select,save=False):
    direct = iso['direct']
    xticklabel = core_names
    fig, ax = plt.subplots(1, 1)
    ind = np.arange(8)
    width = 0.35
    total_porosity_F = list()
    total_porosity_In = list()
    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        total_porosity_F.append(iso[core_name][sample_F]['vpore_total'])
        total_porosity_In.append(iso[core_name][sample_In]['vpore_total'])

    rects1 = ax.bar(ind, total_porosity_In[:], width, color='b')
    rects2 = ax.bar(ind+width+0.02, total_porosity_F[:], width, color='r')
    ax.legend([rects2[0], rects1[0]], ['F', 'In'], fancybox=True, framealpha=0.5)
    plt.xlabel('core names')
    plt.setp(ax, xticks=ind + width + 0.02, xticklabels=xticklabel)
    plt.ylabel('pore volume (cm3/g)')
    if save:
        plt.savefig(direct + 'porosity_all_core_bar.png')
    plt.show()

def plot_isotherm_per_core_in_one_figure(iso,core_names_select,sample_names_select,save=False):

    direct = iso['direct']
    fig, axies = plt.subplots(2, 4, figsize=(14, 8))

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
    # calculate the axis index
        ax_0 = i // 4
        ax_1 = i % 4
        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['p_ads'], iso[core_name][sample_F]['q_ads'], 'r-', markersize=8,
                               linewidth=2,alpha=0.7,label='F')
        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], 'b-', markersize=8,
                               linewidth=2,alpha=0.7,label='In')
        axies[ax_0,ax_1].legend(loc=2)

        axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['p_ads'], iso[core_name][sample_F]['q_ads'],s=80,facecolors='none',linewidths=1,edgecolors='r')
        axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], s=80,
                                  facecolors='none', linewidths=1, edgecolors='b')
        axies[ax_0, ax_1].set_title(core_name)
        axies[ax_0,ax_1].set_xlim([0,1])
        axies[ax_0,ax_1].set_ylim(ymin=0)
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        if ax_1 == 0:
            axies[ax_0,ax_1].set_ylabel('adsorption amount ')
        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')
        fig.text(0.5, 0.04, 'relative pressure', ha='center')
    plt.tight_layout(pad=4,w_pad=2,h_pad=2)

    if save:
        plt.savefig(direct + 'failed_intact_isotherm_8_subplots.png')
    plt.show()

def plot_psd_per_core_in_one_figure(iso,core_names_select,sample_names_select,type ='normal',save=False):
    direct = iso['direct']
    if type == 'log':
        volume_type = 'Vp_dlogD'
    else:
        volume_type ='Vp'

    fig, axies = plt.subplots(2, 4, figsize=(14, 8))

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
    # calculate the axis index
        ax_0 = i // 4
        ax_1 = i % 4
        #iso[core_name][sample_name][] = psd.Davg
        #iso[core_name][sample_name]['Vp'] = psd.Vp
        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['Davg'][1:], iso[core_name][sample_F][volume_type][1:], 'r-', markersize=8,
                               linewidth=2,alpha=0.7,label='F')
        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['Davg'][1:], iso[core_name][sample_In][volume_type][1:], 'b-', markersize=8,
                               linewidth=2,alpha=0.7,label='In')
        axies[ax_0,ax_1].legend(loc=4,fancybox=True, framealpha=0.5)

        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['Davg'], iso[core_name][sample_F][volume_type],s=50,facecolors='none',linewidths=1,edgecolors='k')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['Davg'], iso[core_name][sample_In][volume_type], s=50,facecolors='none', linewidths=1, edgecolors='k')

        axies[ax_0, ax_1].set_title(core_name)
        axies[ax_0,ax_1].set_xscale('log')
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
        fig.text(0.5, 0.04, 'pore size', ha='center')
    plt.tight_layout(pad=4,w_pad=2,h_pad=2)

    if save:
        if type =='log':
            plt.savefig(direct + 'log'+'failed_intact_psd_8_subplots.png')
        else:
            plt.savefig(direct  + 'failed_intact_psd_8_subplots.png')
    plt.show()



def plot_intact_isotherm_all_core_in_one_figure(iso,core_names_select,sample_names_select,save=False):
    direct = iso['direct']
    num_colors = 8
    #cm = plt.get_cmap('Paired')
    color_names = ['black','aqua','crimson','darkorchid','g','b','sienna','grey']
    #print()
    fig, ax = plt.subplots(1, 1)
    #ax.set_color_cycle([cm(1.0 * i / num_colors) for i in range(num_colors)])

    for i, core_name in enumerate(core_names_select):
        # calculate the axis index
        sample_In = sample_names_select[i][1]
        if i <3:
            ax.plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], 'o-', color=color_names[i], markersize=5,
                               linewidth=2,alpha=0.7,label=core_name)
        else:
            ax.plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], '^-',color=color_names[i], markersize=8,
                     linewidth=2, alpha=0.7, label=core_name)

        #ax.imshow(cmap=cm)
        plt.legend(loc=2)

    if save:
        plt.savefig(direct+'all_intact_isotherm.png')
    plt.show()



def plot_isotherm_per_core(iso,core_names,sample_names,save=False):
    direct = iso['direct']
    for i in range(0, len(core_names)):
        core_name = core_names[i]
        figure = plt.figure('ISO_' + core_name)
        legends = []
        for sample_name in sample_names[i]:
            if sample_name[0] == 'I':
                legend, = plt.plot(iso[core_name][sample_name]['p_ads'], iso[core_name][sample_name]['q_ads'], '*-',
                                   markersize=8,
                                   label=iso[core_name]['core_name'] + '_' + iso[core_name][sample_name]['sample_name'])
            else:
                legend, = plt.plot(iso[core_name][sample_name]['p_ads'], iso[core_name][sample_name]['q_ads'], 'o-',
                                   markersize=8,
                                   label=iso[core_name]['core_name'] + '_' + iso[core_name][sample_name]['sample_name'])
            legends.append(legend)
        plt.title('ISO_' + core_name)
        plt.legend(handles=legends, loc=2,fancybox=True, framealpha=0.5)
        plt.xlabel('relative pressure')
        plt.ylabel('adsorption quantity (cm3/g STP)')
        # plt.show()
        if save:
            plt.savefig(direct + 'ISO_' + core_name + '.png')
        plt.show()
        # for i, inputfile in enumerate(input_file_list):

        # data = pd.read_csv(direct+inputfile)
        # data = data.to_dict()
        # print(data)

def plot_porosity_ratio_all_sample(iso,core_names_select,sample_names_select,save=False):
    direct = iso['direct']
    ratio_mesos = []
    ratio_micros = []

    for i, core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        ratio_meso = iso[core_name][sample_F]['vpore_meso'] / iso[core_name][sample_In]['vpore_meso']
        ratio_micro = iso[core_name][sample_F]['vpore_micro'] / iso[core_name][sample_In]['vpore_micro']
        ratio_mesos.append(ratio_meso)
        ratio_micros.append(ratio_micro)
        plt.text(ratio_meso + 0.02, ratio_micro - 0.02, core_name + '_' + sample_F[0], size=10, alpha=0.7)
        if int(core_name[0]) < 3:
            plt.plot(ratio_meso, ratio_micro, 'go', markersize=12)
        else:
            plt.plot(ratio_meso, ratio_micro, 'r^', markersize=12)
    print(ratio_mesos)
    plt.plot([1, 1], [0, 2], 'k--', linewidth=3, alpha=0.4)
    plt.plot([0, 2], [1, 1], 'k--', linewidth=3, alpha=0.4)
    plt.xlim([0.6, 1.5])
    plt.ylim([0.6, 1.5])


    if save:
        plt.savefig(direct + 'porosity_ratio_all_sample.png')
    plt.show()