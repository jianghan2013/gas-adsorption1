from BJH_function import BJH_calculation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from BJH_function import std_slope  # linear fit model



#---------- data reading
def iso_reading_N2(direct,core_names,sample_names):
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

            # read adsorption isotherm
            iso[core_name][sample_name]['filename'] = iso[core_name]['direct']+ 'ISO_'+core_name+'_'+sample_name +'_N2.csv'
            data = pd.read_csv(iso[core_name][sample_name]['filename'])
            iso[core_name][sample_name]['p_ads'] = data['p_ads']
            iso[core_name][sample_name]['q_ads'] = data['q_ads']

            # read full isotherm
            temp_filename = iso[core_name]['direct'] + 'ISO_' + 'full_'+core_name + '_' + sample_name + '_N2.csv'
            data = pd.read_csv(temp_filename)
            iso[core_name][sample_name]['p_full'] = data['p']
            iso[core_name][sample_name]['q_full'] = data['q']


            p_ads = iso[core_name][sample_name]['p_ads']
            q_ads = iso[core_name][sample_name]['q_ads']

            # read psd from 3flex
            filename_psd_3flex = iso[core_name]['direct']+ 'PSD_3flex_'+core_name+'_'+sample_name +'_N2.csv'
            data = pd.read_csv(filename_psd_3flex)
            iso[core_name][sample_name]['Davg_3flex'] = data['Davg_3flex']
            iso[core_name][sample_name]['Vp_dlogD_3flex'] = data['Vp_dlogD_3flex']


            # get porosity from psd 3flex


            # use in-house BJH code to calculate BJH psd
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

            # do fractal number
            fractal_1, fractal_2 = get_fractal_number(p_ads, q_ads, do_plot=False)
            iso[core_name][sample_name]['fractal_1'] = fractal_1
            iso[core_name][sample_name]['fractal_2'] = fractal_2


    return iso


#----------------------- visualization

#--- setting up the default plotting parameter
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
        data_total = list()
        xticklabel = list()
        for sample_name in sample_names[i]:
            data_micro.append(iso[core_name][sample_name]['vpore_micro'])
            data_meso.append(iso[core_name][sample_name]['vpore_meso'])
            data_total.append(iso[core_name][sample_name]['vpore_total'])
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




def plot_BET_surface_bar(df_BET):
    '''

    '''
    #direct = iso['direct']
#    figure = plt.figure
    #matplotlib.rcParams.update(params)

    xticklabel = df_BET['core_name']
    Intact_BET = df_BET['In'][:8]
    HFailed_BET = df_BET['F'][:8]
    VFailed_BET = np.array(df_BET['F'][8:10])
    print(VFailed_BET)
    fig,ax = plt.subplots(1,1)
    ind = np.arange(0,10,10/8)
    width =0.35
    #BET_select
    rects1 = ax.bar(ind, Intact_BET, width, color='b')
    rects2 = ax.bar(ind+width,HFailed_BET, width, color='r')
    rects3 = ax.bar(ind[0]+width+width,VFailed_BET[0],width,color='g')
    rects3 = ax.bar(ind[5] + width + width, VFailed_BET[1], width, color='g')
    #ax.legend()
    ax.set_xlim([-0.2,10])
    plt.setp(ax, xticks=ind + width+0.02, xticklabels=xticklabel)
    ax.legend([rects1[0], rects2[0],rects3[0]], ['Intact', 'HFail','VFail'], fancybox=True, framealpha=0.5)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=20)
    #plt.xlabel('core names')
    plt.ylabel('Specific surface area (m$^2$/g)')
    plt.show()

def plot_total_porosity_bar(iso,core_names_select,sample_names_select,save=False):
    direct = iso['direct']
    xticklabel = core_names_select
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
    params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'legend.fontsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': False,
        'figure.figsize': [14, 18]
    }
    #matplotlib.rcParams.update(params)

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

        #axies[ax_0,ax_1].legend(loc=2)

        print(ax_0,ax_1)


        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_full'], iso[core_name][sample_In]['q_full'], 'bo-', markersize=10,
                               linewidth=2,label='Intact')
        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['p_full'], iso[core_name][sample_F]['q_full'], 'rD-', markersize=10,
                               linewidth=2,label='HFail')
        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso[core_name][sample_VF]['p_full'], iso[core_name][sample_VF]['q_full'], 'g^-',
                                   markersize=10,linewidth=2, label='VFail')
        if (ax_0 == 0) & (ax_1 == 0):
                axies[ax_0, ax_1].legend(loc=2)
        #axies[ax_0, ax_1].legend(loc=2)
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['p_full'], iso[core_name][sample_F]['q_full'],s=50,
        #                          facecolors='r',linewidths=1,edgecolors='r',alpha=1,label='HF')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['p_full'], iso[core_name][sample_In]['q_full'], s=50,
        #                          facecolors='b', linewidths=1, edgecolors='b',alpha=1,label='In')
        #if len(sample_names_select[i]) > 2:
        #    axies[ax_0, ax_1].scatter(iso[core_name][sample_VF]['p_full'], iso[core_name][sample_VF]['q_full'], s=50,
        #                          facecolors='g', linewidths=1, edgecolors='g',alpha=1,label='VF')

        # remove xticks
        if i < 6:
            axies[ax_0,ax_1].set_xticks([])

        #plt.legend(handles=legends)
        if i < 3:
            axies[ax_0, ax_1].set_title('EF_'+core_name,fontsize=20, x=0.6, y=0.8)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + core_name, fontsize=20, x=0.6, y=0.8)
        axies[ax_0,ax_1].set_xlim([-0.05,1.05])
        axies[ax_0,ax_1].set_ylim(ymin=0)
        if i == 4 :
            axies[ax_0, ax_1].set_yticks([0,4,8,12,16])
        elif i == 5 or i ==2:
            axies[ax_0, ax_1].set_yticks([0,4,8,12,16])
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        #if ax_1 == 0:

            #axies[ax_0,ax_1].set_ylabel('adsorption amount ')
        #if ax_1 == 1:
    fig.text(0.05, 0.6, 'Adsorption quantity (cm$^3$/g)', rotation=90)
            #axies[ax_0,ax_1].set_xlabel('a')
    fig.text(0.5, 0.01, 'Relative pressure (p/p$^0$)', ha='center')
    #plt.grid()
    plt.tight_layout(pad=4,w_pad=2,h_pad=0.5)
    #fig.legend((line1,line2,line3),('HF','In','VF'),'upper left')
    if save:
        plt.savefig(direct + 'failed_intact_isotherm_8_subplots.png')
    plt.show()

def plot_psd_per_core_in_one_figure(iso,core_names_select,sample_names_select,type ='3flex',save=False):
    direct = iso['direct']
    if type == 'my':
        # using my own bjh method
        pore_size_type = 'Davg'
        volume_type = 'Vp_dlogD'
    else:
        # using 3flex results
        pore_size_type = 'Davg_3flex'
        volume_type ='Vp_dlogD_3flex'

    fig, axies = plt.subplots(2, 4, figsize=(14, 8))

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
        axies[ax_0, ax_1].plot(iso[core_name][sample_F][pore_size_type][1:], iso[core_name][sample_F][volume_type][1:], 'r-', markersize=8,
                               linewidth=2,alpha=0.7,label='F')
        axies[ax_0, ax_1].plot(iso[core_name][sample_In][pore_size_type][1:], iso[core_name][sample_In][volume_type][1:], 'b-', markersize=8,
                               linewidth=2,alpha=0.7,label='In')


        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso[core_name][sample_VF][pore_size_type][1:], iso[core_name][sample_VF][volume_type][1:],
                                   'g-', markersize=8,
                                   linewidth=2, alpha=0.7, label='VF')
        axies[ax_0, ax_1].legend(loc=2, fancybox=True, framealpha=0.5)
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['Davg'], iso[core_name][sample_F][volume_type],s=50,facecolors='none',linewidths=1,edgecolors='k')
        #axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['Davg'], iso[core_name][sample_In][volume_type], s=50,facecolors='none', linewidths=1, edgecolors='k')

        axies[ax_0, ax_1].set_title(core_name)
        axies[ax_0,ax_1].set_xscale('log')
        axies[ax_0,ax_1].set_xlim([10,3000])
        axies[ax_0,ax_1].set_ylim(ymin=0)
        #axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'],'o-',markersize=8,linewidth=2)
        if ax_1 == 0:

                axies[ax_0,ax_1].set_ylabel('dv/dlog pore volume')

        #if ax_1 == 1:
            #axies[ax_0,ax_1].set_xlabel('a')
    fig.text(0.5, 0.04, 'pore size', ha='center')
    plt.tight_layout(pad=4,w_pad=2,h_pad=2)

    if save:
        if type =='my':
            plt.savefig(direct + 'myBJH_'+'failed_intact_psd_8_subplots.png')
        else:
            plt.savefig(direct  + '3flex_' 'failed_intact_psd_8_subplots.png')
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
        ratio_total = iso[core_name][sample_F]['vpore_total'] / iso[core_name][sample_In]['vpore_total']
        ratio_mesos.append(ratio_meso)
        ratio_micros.append(ratio_micro)
        ratio_total.append(ratio_total)
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
    return ratio_mesos,ratio_micros

def plot_df(df,col1,col2,xlim=[0.8,1.8,1],ylim=[0.6,1.8,1],linear = False,x_inter=[0.9,1.4],xylabel=['','']):
    # xlim the start and end of xlim x[2] determine whether or not plot the line

    plt.plot(df[col1][:3], df[col2][:3], 'ro', alpha=1,label='HFail EF')
    plt.plot(df[col1][3:8], df[col2][3:8], 'go', alpha=1,label='HFail NRM')

    plt.plot(df[col1][8:9], df[col2][8:9], 'r^',  alpha=1,label='VFail EF')
    plt.plot(df[col1][9:10], df[col2][9:10], 'g^',  alpha=1,label='VFail NRM ')
    #plt.legend(loc=2,fancybox=True, framealpha=0.5)
    if xylabel[0] != '':
        plt.xlabel(xylabel[0])
        plt.ylabel(xylabel[1])
    if ylim[2] is not None:
        plt.plot([ylim[2], ylim[2]], [ylim[0], ylim[1]], 'k--', linewidth=3, alpha=0.4)
    if xlim[2] is not None:
        plt.plot([xlim[0], xlim[1]], [xlim[2], xlim[2]], 'k--', linewidth=3, alpha=0.4)


    if xlim[0] is not None:
        plt.xlim(xlim[0:2])
        plt.ylim(ylim[0:2])

    #plt.text(df[col1][3], df[col2][3], df.index.values[3])
    #plt.text(df[col1][6], df[col2][6], df.index.values[6])
    #plt.text(df[col1][0],df[col2][0],df.index.values[0])
    #plt.text(df[col1][5], df[col2][5], df.index.values[5])
    #plt.text(df[col1][8] , df[col2][8] , df.index.values[8])
    #plt.text(df[col1][9], df[col2][9] , df.index.values[9])
    #plt.text(df[col1] + 0.02, df[col2] - 0.02, df.index.values, size=10, alpha=0.7)


    plt.grid()
    if linear:
        reg_EF = linear_model.LinearRegression()
        #a= (1,2,3)
        reg_NRM = linear_model.LinearRegression()

        x_NRM_train = np.array(df[col1][np.r_[3:8,9:10]]).reshape(-1,1)
        y_NRM_train = np.array(df[col2][np.r_[3:8,9:10]]).reshape(-1,1)

        x_EF_train = np.array(df[col1][np.r_[0:3,8:9]]).reshape(-1,1)
        y_EF_train = np.array(df[col2][np.r_[0:3,8:9]]).reshape(-1,1)
        #print(x_EF_train.shape,y_EF_train.shape)
        reg_EF.fit(x_EF_train,y_EF_train)
        reg_NRM.fit(x_NRM_train,y_NRM_train)

        r2_EF = round( r2_score(y_EF_train,reg_EF.predict(x_EF_train)) , 3)
        r2_NRM = round( r2_score(y_NRM_train, reg_NRM.predict(x_NRM_train)), 3)

        EF_slope = round(reg_EF.coef_[0][0],3)
        NRM_slope = round(reg_NRM.coef_[0][0],3)

        x_EF_test = np.array( [x_inter[0],x_inter[1]] ).reshape(-1,1)
        y_EF_test_hat = reg_EF.predict(np.array(x_EF_test))
        x_NRM_test = np.array( [ x_inter[0],x_inter[1] ]  ).reshape(-1, 1)

        y_NRM_test_hat = reg_NRM.predict(np.array(x_NRM_test))
        plt.text(x_EF_test[-1]*1.01,y_EF_test_hat[-1],'k= '+ str(EF_slope)+', R$^2$='+str(r2_EF))
        plt.text(x_NRM_test[-1]*1.01, y_NRM_test_hat[-1], 'k= ' + str(NRM_slope)+', R$^2$='+ str(r2_NRM))
        plt.plot(x_EF_test, y_EF_test_hat,'r-',linewidth=2)
        plt.plot(x_NRM_test, y_NRM_test_hat, 'g-',linewidth=2)
    #plt.grid
    plt.show()

# ----- plot the mineral composition bar
def plot_mineral_bar(df,keys =['calcite','illite/mica','mixture illite/smectite','quartz','plagioclase','pyrite'],save =False):
    import operator

    matplotlib.rcParams.update(params)

    #file_prechar ='C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/precharacterization_data.xlsx'
    #df_xrd = pd.read_excel(file_prechar,sheetname='xrd')


    fig,axis = plt.subplots(1,1)

    width =0.8
    ind = np.arange(8)
    rects = []
    bottom = tuple([0]*8)
    print(bottom)
    color_names = [ 'orange', 'crimson', 'darkorchid', 'g', 'b', 'black','sienna', 'grey']
    for i,key in enumerate(keys):
        print(i)
        if i > 0:
            a = tuple((df[keys[i - 1]][0:8]))
            bottom = tuple(map(operator.add, a, bottom))
        rect = axis.bar(ind,df[key][0:8],width,bottom=bottom,color=color_names[i])
        rects.append(rect)
    # add 'other' as the last mineral
    a = tuple((df[keys[-1]][0:8]))
    bottom = tuple(map(operator.add, a, bottom))
    rect = axis.bar(ind,100 - np.array(bottom),width,bottom=bottom,color=color_names[i+1])
    rects.append(rect)

    plt.setp(axis,xticks=ind+width/2,xticklabels=df['core_name'][0:8])
        #axies[ax_0,ax_1].set_xticks(xticklabel)
    keys.append('other')
    #axis.legend(rects,keys,fancybox=True, )
    box = axis.get_position()
    axis.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.ylabel('Mineral concentration (wt. %)')
    axis.legend(rects,keys,loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=4,framealpha=1)
    axis.set_xticklabels(axis.xaxis.get_majorticklabels(), rotation=20)

    plt.show()

#---- plot BET C value
def plot_BET_constant(df,col):
    a = np.arange(8)
    fig,ax = plt.subplots()
    ax.plot(a[:3],df[col][:3],'ro',markersize=10,alpha=0.8)
    ax.plot([-1, 8], [100, 100], 'k--', linewidth=1)
    ax.plot([-1,8],[0,0],'k--',linewidth=1)
    ax.plot(a[3:8], df[col][3:8], 'go',markersize=10,alpha=0.8)
    #ax.set_yscale('symlog')
    plt.xlim([-0.5,7.5])
    plt.ylim([-800,600])
    plt.show()


#---------- plot full isotherm
def plot_full_N2_isotherm():
    direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/'
    filename = direct + 'isotherm_N2.xlsx'
    df_3_53 = pd.read_excel(filename,sheetname = '3_53')
    df_1_223 = pd.read_excel(filename,sheetname = '1_223')
    plt.figure()
    #plt.plot(df_3_53['p'][0:30],df_3_53['q'][0:30],'ro-',markersize=10)
    #plt.plot(df_3_53['p'][29:],df_3_53['q'][29:],'ro-',markersize=10,alpha=0.5)

    l2,=plt.plot(df_3_53['p'][29:], df_3_53['q'][29:], 'go-',markersize=10,linewidth=1)
    l1, = plt.plot(df_3_53['p'][:30], df_3_53['q'][:30], 'ro-', markersize=10, linewidth=1)
    plt.legend([l1, l2], ['adsorption', 'desorption'],loc=2)
    plt.title('3_53')
    plt.xlabel('relative pressure (p/p0)')
    plt.ylabel('adsorption quantity (cm3/g)')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.5, 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    plt.figure()
    #fig, ax = plt.subplots()
    l2,=plt.plot(df_1_223['p'][27:], df_1_223['q'][27:], 'go-',linewidth=1,markersize=10,label='desorption')
    l1, = plt.plot(df_1_223['p'][:28], df_1_223['q'][:28], 'ro-', linewidth=1, markersize=10, label='adsorption')
    #plt.scatter(df_1_223['p'][:28], df_1_223['q'][:28], s=80, facecolors='g', edgecolors='g', linewidths=1)
    #plt.scatter(df_1_223['p'][27:], df_1_223['q'][27:], s=80, facecolors='none', edgecolors='g', linewidths=1)
    #plt.scatter(df_3_53['p'][29:], df_3_53['q'][29:], s=80, facecolors='none', edgecolors='r', linewidths=1)
    #print(df_1_223['p'][27])
    #plt.plot(,'ro-',markersize=10,alpha=0.5)
    #q1 =
    #print(p1[29])
    plt.title('1_223',fontsize=16)
    plt.xlabel('relative pressure (p/p0)',fontsize=16)
    plt.ylabel('adsorption quantity (cm3/g)',fontsize=16)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.5,12)
    plt.legend([l1,l2],['adsorption','desorption'],loc=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_full_CO2_isotherm():
    direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/'
    filename = direct + 'isotherm_co2.xlsx'
    df_3_53 = pd.read_excel(filename, sheetname='3_53')
    df_1_223 = pd.read_excel(filename, sheetname='1_223')

    plt.figure()
    l1, = plt.plot(df_3_53['p'], df_3_53['q'], 'ro-', linewidth=1, markersize=10, label='adsorption')
    plt.legend([l1], ['adsorption'], loc=4)
    plt.title('3_53', fontsize=16)
    plt.xlabel('relative pressure (p/p0)', fontsize=16)
    plt.ylabel('adsorption quantity (cm3/g)', fontsize=16)
    plt.xlim(-0.0005,0.025)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    plt.figure()
    l1, = plt.plot(df_1_223['p'], df_1_223['q'], 'ro-', linewidth=1, markersize=10, label='adsorption')
    plt.legend([l1], ['adsorption'], loc=4)
    plt.title('1_223', fontsize=16)
    plt.xlabel('relative pressure (p/p0)', fontsize=16)
    plt.ylabel('adsorption quantity (cm3/g)', fontsize=16)
    plt.xlim(-0.0005,0.025)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_fractal_dimension_method(iso):
    p = iso['3_53']['In_1']['p_ads']
    q = iso['3_53']['In_1']['q_ads']
    get_fractal_number(p, q, do_plot=True)

def plot_BET_surface(iso):
    p  = iso['3_53']['In_1']['p_ads']
    q  = iso['3_53']['In_1']['q_ads']
    get_BET_surface(p,q,do_plot=True)

#--------------- calculation
def get_porosity_ratio_all_sample_BJH(iso,core_names_select,sample_names_select):
    direct = iso['direct']
    ratio_mesos = []
    ratio_micros = []
    ratio_totals = []


    for i, core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]

        ratio_meso = iso[core_name][sample_F]['vpore_meso'] / iso[core_name][sample_In]['vpore_meso']
        ratio_micro = iso[core_name][sample_F]['vpore_micro'] / iso[core_name][sample_In]['vpore_micro']
        ratio_total = iso[core_name][sample_F]['vpore_total'] / iso[core_name][sample_In]['vpore_total']

        ratio_mesos.append(ratio_meso)
        ratio_micros.append(ratio_micro)
        ratio_totals.append(ratio_total)

    print(ratio_mesos)


    #plt.show()
    return ratio_mesos,ratio_micros,ratio_totals

def get_porosity_ratio_all_sample_3flex(iso,core_names_select,sample_names_select):
    direct = iso['direct']
    ratio_mesos = []
    ratio_micros = []
    ratio_totals = []


    for i, core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]

        ratio_meso = iso[core_name][sample_F]['vpore_meso'] / iso[core_name][sample_In]['vpore_meso']
        ratio_micro = iso[core_name][sample_F]['vpore_micro'] / iso[core_name][sample_In]['vpore_micro']
        ratio_total = iso[core_name][sample_F]['vpore_total'] / iso[core_name][sample_In]['vpore_total']

        ratio_mesos.append(ratio_meso)
        ratio_micros.append(ratio_micro)
        ratio_totals.append(ratio_total)

    print(ratio_mesos)


    #plt.show()
    return ratio_mesos,ratio_micros,ratio_totals

def save_fractal_number(iso,core_names,sample_names,do_save=False):
    # to save the fractal number into dataframe after iso

    data = {}
    data['name_core']=[]
    data['name_sample']=[]
    data['D11']=[]
    data['D12'] = []
    data['D21'] = []
    data['D22'] = []
    data['k1'] = []
    data['k2'] = []
    data['r2_1'] = []
    data['r2_2'] = []
    key1 = ['D11','D12','k1','r2_1']
    key2 = ['D21','D22','k2','r2_2']
    #core_name = '3_42'
    #sample_name = 'In_2'
    for i, core_name in enumerate(core_names):
        for sample_name in sample_names[i]:
            data['name_core'].append(core_name)
            data['name_sample'].append(sample_name)
            fractal_1 = iso[core_name][sample_name]['fractal_1']
            fractal_2 = iso[core_name][sample_name]['fractal_2']
            for key in key1:
                data[key].append(fractal_1[key])
            for key in key2:
                data[key].append(fractal_2[key])
            #data.update(fractal_2)
            #print(fractal_1)
    df = pd.DataFrame(data=data)
    print(df)
    if do_save:
        df.to_csv(iso['direct']+'fractal.csv',index=False)





def get_fractal_number(p,q,p1_end=[0.05,0.45],p2_end=[0.45,1.0],do_plot=False):
    '''


    :param p: adsorption branch
    :param q: adsorption branch
    :param p1_end: end point for fractal 1 interval
    :param p2_end: end point for fractal 2 interval
    :param do_plot:
    :return: fractal 1: p <  0.45
            fractal 2: p >0.45
    '''

    # select the pressure range
    ind1 = ( p <= p1_end[1] ) & ( p > p1_end[0])
    ind2 = (p > p2_end[0])  & (p < p2_end[1])

    # select the pressure and quantity in that range
    p1 = p[ind1]
    q1 = q[ind1]
    p2 = p[ind2]
    q2 = q[ind2]

    # prepare for x and y
    #reg_1 = linear_model.LinearRegression()
    x1 = np.log(np.log(1 / p1)).reshape(-1, 1)
    y1 = np.log(q1).reshape(-1, 1)
    x2 = np.log(np.log(1 / p2)).reshape(-1, 1)
    y2 = np.log(q2).reshape(-1, 1)

    # model fitting
    my_model1 = std_slope.get_linear_model(x1, y1)
    my_model1.linear_fit()

    my_model2 = std_slope.get_linear_model(x2, y2)
    my_model2.linear_fit()

    # get fractal dimension coefficients
    k1 = round(my_model1.coeffs['slope'], 4)
    k2 = round(my_model2.coeffs['slope'], 4)

    Dzone1_1 = round(3 * k1 + 3, 4)
    Dzone1_2 = round(k1 + 3, 4)
    Dzone2_1 = round(3 * k2 + 3, 4) # zone 2 dimension 1 fractal dimension zone 2 is for p/p0 > 0.45
    Dzone2_2 = round(k2 + 3, 4) # zone 2 dimension 2

    # get accuracy coefficients R2 and standard error of slope
    R2_1 = round(my_model1.coeffs['R2'], 4)
    R2_2 = round(my_model2.coeffs['R2'], 4)
    SE_D22 = round(my_model2.coeffs['standard_error_slope'], 4)
    SE_D21 = round(my_model2.coeffs['standard_error_slope'], 4)*3

    SE_D12 = round(my_model1.coeffs['standard_error_slope'], 4)
    SE_D11 = round(my_model1.coeffs['standard_error_slope'], 4) * 3


    # for plotting


    if do_plot:
        # print(k2,r2_2)
        # print(k1, r2_1)
        x1_test = x1[np.r_[0, -1]]
        y1_hat = my_model1.regression_model.predict(x1_test)
        x2_test = x2[np.r_[0, -1]]
        y2_hat = my_model2.regression_model.predict(x2_test)

        figure = plt.figure()
        plt.plot(x2, y2, 'ro', markersize=10)
        plt.plot(x1, y1, 'go', markersize=10)
        plt.plot(x2_test, y2_hat, 'k', linewidth=2)
        plt.plot(x1_test, y1_hat, 'k', linewidth=2)
        plt.plot([-.22501, -.22501], [0, 3], 'k--', linewidth=3)
        plt.xlabel('ln(ln(p0/p))')
        plt.ylabel('ln(V)')
        plt.title('Fractal FFH method')
        plt.show()

    # output results
    fractal_zone2 = {}
    fractal_zone2['D21'] = Dzone2_1
    fractal_zone2['D22'] = Dzone2_2
    fractal_zone2['k2'] = k2
    fractal_zone2['R2_2'] = R2_2
    fractal_zone2['SE_D22'] = SE_D22
    fractal_zone2['SE_D21'] = SE_D21

    fractal_zone1 = {}
    fractal_zone1['D11'] = Dzone1_1
    fractal_zone1['D12'] = Dzone1_2
    fractal_zone1['k1'] = k1
    fractal_zone1['R2_1'] = R2_1
    fractal_zone1['SE_D11'] = SE_D11
    fractal_zone1['SE_D12'] = SE_D12

    return fractal_zone1,fractal_zone2


def get_BET_surface(p,q,do_plot=False):
    # has some problem
    ind = (p >= 0.05) & ( p <= 0.35 )
    p1 = p[ind]
    q1 = q[ind]
    reg = linear_model.LinearRegression()
    x_train =  np.array(p1).reshape(-1,1)
    y_train =  np.array( 1.0/(q1*(1/p1) -1 )  ).reshape(-1,1)
    reg.fit(x_train, y_train)

    x_test = x_train[np.r_[0, -1]]
    y_hat = reg.predict(x_test)
    slope = round(reg.coef_[0][0], 4) # slope
    intersect =  reg.intercept_[0]
    BET_const = 1 + slope/intersect
    BET_Qm = 1/( BET_const*intersect)
    print(slope,intersect)
    print(BET_Qm,BET_const)
    if do_plot:
        # print(k2,r2_2)
        # print(k1, r2_1)
        figure = plt.figure()
        plt.plot(x_train, y_train, 'go', markersize=10)
        plt.plot(x_test, y_hat, 'k', linewidth=2)
        #plt.plot([-.22501, -.22501], [0, 3], 'k--', linewidth=3)
        plt.xlabel('p/p0')
        plt.ylabel('1/ q(po/p-1)')
        plt.title('BET surface area')
        plt.show()






