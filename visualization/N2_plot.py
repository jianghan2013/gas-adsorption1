from BJH_function import BJH_calculation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import r2_score



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

            # read isotherm
            iso[core_name][sample_name]['filename'] = iso[core_name]['direct']+ 'ISO_'+core_name+'_'+sample_name +'_N2.csv'
            data = pd.read_csv(iso[core_name][sample_name]['filename'])
            iso[core_name][sample_name]['p_ads'] = data['p_ads']
            iso[core_name][sample_name]['q_ads'] = data['q_ads']

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

    direct = iso['direct']
    fig, axies = plt.subplots(2, 4, figsize=(14, 8))

    for i,core_name in enumerate(core_names_select):
        sample_F = sample_names_select[i][0]
        sample_In = sample_names_select[i][1]
        if len(sample_names_select[i]) > 2:
            sample_VF = sample_names_select[i][2]
    # calculate the axis index
        ax_0 = i // 4
        ax_1 = i % 4

        #axies[ax_0,ax_1].legend(loc=2)


        axies[ax_0, ax_1].plot(iso[core_name][sample_F]['p_ads'], iso[core_name][sample_F]['q_ads'], 'r-', markersize=8,
                               linewidth=2,label='HF')

        axies[ax_0, ax_1].plot(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], 'b-', markersize=8,
                               linewidth=2,label='In')
        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].plot(iso[core_name][sample_VF]['p_ads'], iso[core_name][sample_VF]['q_ads'], 'g-',
                                   markersize=8,linewidth=2, label='VF')
        axies[ax_0, ax_1].legend(loc=2)
        axies[ax_0, ax_1].scatter(iso[core_name][sample_F]['p_ads'], iso[core_name][sample_F]['q_ads'],s=80,
                                  facecolors='none',linewidths=1,edgecolors='r',alpha=0.4,label='HF')
        axies[ax_0, ax_1].scatter(iso[core_name][sample_In]['p_ads'], iso[core_name][sample_In]['q_ads'], s=80,
                                  facecolors='none', linewidths=1, edgecolors='b',alpha=0.4,label='In')
        if len(sample_names_select[i]) > 2:
            axies[ax_0, ax_1].scatter(iso[core_name][sample_VF]['p_ads'], iso[core_name][sample_VF]['q_ads'], s=80,
                                  facecolors='none', linewidths=1, edgecolors='g',alpha=0.4,label='VF')





        #plt.legend(handles=legends)
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

def plot_df(df,col1,col2,plotline=False,linear = False):
    plt.plot(df[col1][:3],df[col2][:3],'ro',markersize=10,alpha=0.8)
    plt.plot(df[col1][3:8], df[col2][3:8], 'go',markersize=10,alpha=0.8)
    plt.plot(df[col1][8:9], df[col2][8:9], 'r^',markersize=10,alpha=0.8)
    plt.plot(df[col1][9:10], df[col2][9:10], 'g^',markersize=10,alpha=0.8)

    plt.xlabel(col1)
    plt.ylabel(col2)
    if plotline:
        plt.plot([1, 1], [0.8, 1.6], 'k--', linewidth=3, alpha=0.4)
        plt.plot([0.8, 1.6], [1, 1], 'k--', linewidth=3, alpha=0.4)
        plt.xlim([0.6, 1.8])
        plt.ylim([0.6, 1.8])
    plt.text(df[col1][3], df[col2][3]*0.97, df.index.values[3])
    plt.text(df[col1][6], df[col2][6]*0.97, df.index.values[6])
    plt.text(df[col1][0]*0.98,df[col2][0]*1.05,df.index.values[0])
    plt.text(df[col1][5]*0.98, df[col2][5]*0.95, df.index.values[5])
    plt.text(df[col1][8]*0.98 , df[col2][8] * 1.05, df.index.values[8])
    plt.text(df[col1][9]*0.95, df[col2][9] * 0.95, df.index.values[9])
    #plt.text(df[col1] + 0.02, df[col2] - 0.02, df.index.values, size=10, alpha=0.7)

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

        x_EF_test = np.array( [-0.002, 0.004] ).reshape(-1,1)
        y_EF_test_hat = reg_EF.predict(np.array(x_EF_test))
        x_NRM_test = np.array( [ -0.002, 0.004  ]  ).reshape(-1, 1)

        y_NRM_test_hat = reg_NRM.predict(np.array(x_NRM_test))
        plt.text(x_EF_test[-1],y_EF_test_hat[-1],'k= '+ str(EF_slope)+',R^2='+str(r2_EF))
        plt.text(x_NRM_test[-1], y_NRM_test_hat[-1], 'k= ' + str(NRM_slope)+',R^2='+ str(r2_NRM))
        plt.plot(x_EF_test, y_EF_test_hat,'r-',linewidth=2)
        plt.plot(x_NRM_test, y_NRM_test_hat, 'g-',linewidth=2)



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

def get_all_fractal_number(iso):
    pass

def get_fractal_number(p,q,do_plot=False):
     ind1 = ( p <= 0.45 ) & ( p > 0.036148)
     ind2 = (p > 0.45) #& (p < 0.993284)
     p1 = p[ind1]
     q1 = q[ind1]
     p2 = p[ind2]
     q2 = q[ind2]

     #1
     reg_1 = linear_model.LinearRegression()
     x1 = np.log(np.log(1 / p1)).reshape(-1, 1)
     y1 = np.log(q1).reshape(-1, 1)

     reg_1.fit(x1, y1)
     r2_1 = round(r2_score(y1, reg_1.predict(x1)), 4)
     k1 = round(reg_1.coef_[0][0], 4)

     x1_test = x1[np.r_[0, -1]]
     y1_hat = reg_1.predict(x1_test)

     # 2
     reg_2= linear_model.LinearRegression()
     x2 = np.log( np.log(1/p2) ).reshape(-1,1)
     y2 = np.log( q2 ).reshape(-1,1)

     reg_2.fit(x2, y2)
     r2_2 = round(r2_score(y2, reg_2.predict(x2)), 4)
     k2 = round(reg_2.coef_[0][0], 4)

     x2_test = x2[np.r_[0,-1]]
     y2_hat = reg_2.predict(x2_test)

     Dzone2_1 = 3*k2+3
     Dzone2_2 = k2+3
     Dzone1_1 = 3*k1+3
     Dzone1_2 = k1+3
     print(Dzone2_1,Dzone2_2,r2_2)
     print(Dzone1_1,Dzone1_2,r2_1)

     if do_plot:
        print(k2,r2_2)
        print(k1, r2_1)
        figure = plt.figure()
        plt.plot(x2,y2,'ro')
        plt.plot(x1, y1, 'go')
        plt.plot(x2_test,y2_hat,linewidth=2)
        plt.plot(x1_test, y1_hat,linewidth=2)

        plt.show()





