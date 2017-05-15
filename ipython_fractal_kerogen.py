import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from visualization import N2_plot
from BJH_function import BJH_calculation
import matplotlib

def get_clay_isotherm(p_shale,q_shale,p_kerogen,q_kerogen):
    spline_kerogen = BJH_calculation.get_spline(p_kerogen, q_kerogen, 2)
    # spline interpolation
    q_inter_kerogen = spline_kerogen(p_shale)
    q_inter_clay = q_shale - q_inter_kerogen
    return q_inter_clay,q_inter_kerogen

# calculat the fractal dimension
def get_fractal_dimension(i=7,do_plot=False,do_save=False):
    # i between 0-7

    root_name = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/'
    shale_filenames = ['ISO_1_223_In_1_N2','ISO_2_50_In_1_N2','ISO_2_93_In_1_N2',
                       'ISO_3_14_In_1_N2','ISO_3_42_In_1_N2','ISO_3_53_In_1_N2',
                       'ISO_4_14_In_1_N2','ISO_4_34_In_3_N2'
                       ]
    shale_filename = shale_filenames[i]
    df_shale = pd.read_csv(root_name+shale_filename+'.csv')
    p_shale,q_shale = np.array(df_shale['p_ads']), np.array(df_shale['q_ads'])

    # read the kerogen file
    filename_kerogen = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/kerogen_N2.xlsx'
    sheetnames = ['ke_1_223','ke_2_50','ke_2_93',
                  'ke_3_14','ke_3_42','ke_3_53',
                  'ke_4_14','ke_4_34'
                  ]
    sheetname = sheetnames[i]
    df_kerogen = pd.read_excel(filename_kerogen,sheetname)


    p_kerogen,q_kerogen = np.array(df_kerogen['p']), np.array(df_kerogen['q_bulk'])

    # interpolate to get clay
    q_inter_clay, q_inter_kerogen = get_clay_isotherm(p_shale,q_shale,p_kerogen,q_kerogen)

    # calculate the fractal dimension
    p2_end = [0.45,1.0]
    _, f2_kerogen = N2_plot.get_fractal_number(p_kerogen,q_kerogen,p2_end = p2_end,do_plot=False)
    _, f2_inter_kerogen = N2_plot.get_fractal_number(p_shale,q_inter_kerogen,p2_end = p2_end,do_plot=False)
    if i>2:
        _, f2_clay = N2_plot.get_fractal_number(p_shale,q_inter_clay,p2_end = p2_end,do_plot=False)
    _, f2_shale = N2_plot.get_fractal_number(p_shale,q_shale,p2_end = p2_end,do_plot=False)
    print(shale_filename)
    #print(f2_shale['D22'],'\t',f2_kerogen['D22'],'\t',f2_clay['D22'])
    print('name\t\tD22\t\tSE\t\t\tR2\t')
    print('shale\t',f2_shale['D22'],'\t',f2_shale['SE_D22'],'\t',f2_shale['R2_2'])
    print('kerogen\t',f2_kerogen['D22'],'\t',f2_kerogen['SE_D22'],'\t',f2_kerogen['R2_2'])
    if i>2:
        print('clay\t',f2_clay['D22'],'\t',f2_clay['SE_D22'],'\t',f2_clay['R2_2'])
    print('kerogen_porosity/shale_porosity\t',q_inter_kerogen[-1]/q_shale[-1])
    #print(f2_kerogen['D22'],'\t')
    #print(f2_inter_kerogen['D22'])
    #print(q_inter_kerogen[-1]/q_shale[-1])

    if do_plot:
    #plt.plot(p_kerogen,q_kerogen,'o')
        plt.plot(p_shale,q_shale,'ro-')
        plt.plot(p_shale,q_inter_kerogen,'go')
        plt.plot(p_shale,q_inter_clay,'ko')
        plt.plot([0,1.0],[0,0],'k--',linewidth=3)
        plt.title(shale_filename)
        plt.show()

    # save the file
    clay_filenames = ['clay_1_223_N2','clay_2_50_N2','clay_2_93_N2',
                       'clay_3_14_N2','clay_3_42_N2','clay_3_53_N2',
                       'clay_4_14_N2','clay_4_34_N2'
                       ]

    if do_save:
        data = {'p':p_shale,'q':q_inter_clay}
        df_clay = pd.DataFrame(data)
        df_clay.to_csv(root_name+clay_filenames[i]+'.csv',index=False)
        print('saved')


def read_full_isotherm(i=0):
    # i between 0-7

    # read shale
    root_name = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/'
    shale_filenames = ['ISO_full_1_223_In_1_N2', 'ISO_full_2_50_In_1_N2', 'ISO_full_2_93_In_1_N2',
                       'ISO_full_3_14_In_1_N2', 'ISO_full_3_42_In_1_N2', 'ISO_full_3_53_In_1_N2',
                       'ISO_full_4_14_In_1_N2', 'ISO_full_4_34_In_3_N2'
                       ]
    shale_filename = shale_filenames[i]
    df_shale = pd.read_csv(root_name + shale_filename + '.csv')
    p_shale, q_shale = np.array(df_shale['p']), np.array(df_shale['q'])

    # read the kerogen file
    filename_kerogen = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/kerogen_full_N2.xlsx'
    sheetnames = ['ke_1_223', 'ke_2_50', 'ke_2_93',
                  'ke_3_14', 'ke_3_42', 'ke_3_53',
                  'ke_4_14', 'ke_4_34'
                  ]
    sheetname = sheetnames[i]
    df_kerogen = pd.read_excel(filename_kerogen, sheetname)
    p_kerogen, q_kerogen = np.array(df_kerogen['p']), np.array(df_kerogen['q_bulk'])

    # read the clay
    clay_filenames = ['clay_1_223_N2', 'clay_2_50_N2', 'clay_2_93_N2',
                      'clay_3_14_N2', 'clay_3_42_N2', 'clay_3_53_N2',
                      'clay_4_14_N2', 'clay_4_34_N2'
                      ]
    clay_filename = clay_filenames[i]
    df_clay = pd.read_csv(root_name + clay_filename + '.csv')
    p_clay, q_clay = np.array(df_clay['p']), np.array(df_clay['q'])

    isotherms ={'p_shale':p_shale,'q_shale':q_shale,
               'p_kerogen': p_kerogen,'q_kerogen':q_kerogen,
               'p_clay': p_clay, 'q_clay':q_clay
               }
    return isotherms

def read_full_psd(i=0):
    # i between 0-7

    # read shale
    root_name = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/'
    shale_filenames = ['PSD_3flex_1_223_In_1_N2', 'PSD_3flex_2_50_In_1_N2', 'PSD_3flex_2_93_In_1_N2',
                       'PSD_3flex_3_14_In_1_N2', 'PSD_3flex_3_42_In_1_N2', 'PSD_3flex_3_53_In_1_N2',
                       'PSD_3flex_4_14_In_1_N2', 'PSD_3flex_4_34_In_3_N2'
                       ]
    shale_filename = shale_filenames[i]
    df_shale = pd.read_csv(root_name + shale_filename + '.csv')
    Davg_shale, Vp_dlogD_shale = np.array(df_shale['Davg_3flex']), np.array(df_shale['Vp_dlogD_3flex'])

    # read the kerogen file
    filename_kerogen = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/kerogen/psd_kerogen_N2.xlsx'
    sheetnames = ['ke_1_223', 'ke_2_50', 'ke_2_93',
                  'ke_3_14', 'ke_3_42', 'ke_3_53',
                  'ke_4_14', 'ke_4_34'
                  ]
    sheetname = sheetnames[i]
    df_kerogen = pd.read_excel(filename_kerogen, sheetname)
    Davg_kerogen, Vp_dlogD_kerogen = np.array(df_kerogen['Davg_3flex']), np.array(df_kerogen['Vp_dlogD_3flex_bulk'])

    # read the clay


    PSDs ={'Davg_shale':Davg_shale,'Vp_dlogD_shale':Vp_dlogD_shale,
               'Davg_kerogen': Davg_kerogen,'Vp_dlogD_kerogen':Vp_dlogD_kerogen,
               }
    return PSDs

def plot_isotherm_per_core():
    sample = []
    names =  ['1_223', '2_50', '2_93',
                      '3_14', '3_42', '3_53',
                      '4_14', '4_34'
                 ]
    for i in range(8):
        isotherms = read_full_isotherm(i)
        sample.append(isotherms)

    fig, axies = plt.subplots(4, 2)
    for i in range(8):
        # calculate the axis index
        ax_0 = i // 2
        ax_1 = i % 2

        # axies[ax_0,ax_1].legend(loc=2)

        print(ax_0, ax_1)

        axies[ax_0, ax_1].plot(sample[i]['p_shale'],sample[i]['q_shale'],'bo-', markersize=8,
                               linewidth=2, alpha=1, label='Shale')
        axies[ax_0, ax_1].plot(sample[i]['p_kerogen'],sample[i]['q_kerogen'],'y^-', markersize=8,
                               linewidth=2, alpha=1, label='Kerogen')
        axies[ax_0, ax_1].plot(sample[i]['p_clay'],sample[i]['q_clay'],'D-',color='darkorchid', markersize=8,
                               linewidth=2, alpha=1, label='Clay')
        #axies[ax_0, ax_1].legend(loc=2, )

        if (ax_0 == 0) & (ax_1 == 0):
            axies[ax_0, ax_1].legend(loc=2,fancybox=True, framealpha=0.5)
        if i < 6:
            axies[ax_0, ax_1].set_xticks([])

        # plt.legend(handles=legends)
        if i < 3:
            axies[ax_0, ax_1].set_title('EF_' + names[i], fontsize=20, x=0.65, y=0.8)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + names[i], fontsize=20, x=0.65, y=0.8)
        axies[ax_0, ax_1].set_xlim([-0.05, 1.05])
        axies[ax_0, ax_1].set_ylim(ymin=-1)
        if i == 4 or i== 3 or i ==2 :
            axies[ax_0, ax_1].set_yticks([0, 4, 8, 12, 16])
        #elif i == 5 :
        #    axies[ax_0, ax_1].set_yticks([0, 2, 8, 12, 14])
    fig.text(0.05, 0.6, 'Adsorption quantity (cm$^3$/g)', rotation=90)
    # axies[ax_0,ax_1].set_xlabel('a')
    fig.text(0.5, 0.01, 'Relative pressure (p/p$^0$)', ha='center')
    # plt.grid()
    plt.tight_layout(pad=4, w_pad=2, h_pad=0.5)
    # fig.legend((line1,line2,line3),('HF','In','VF'),'upper left')
    plt.show()

#------- plot psd per core with kerogen, bulk shale psd
def plot_psd_per_core():
    sample = []
    ylim_max = [0.015, 0.017, 0.02, 0.02, 0.018, 0.016, 0.012, 0.014]
    names =  ['1_223', '2_50', '2_93',
                      '3_14', '3_42', '3_53',
                      '4_14', '4_34'
                 ]
    for i in range(8):
        PSDs = read_full_psd(i)
        sample.append(PSDs)

    fig, axies = plt.subplots(4, 2)
    for i in range(8):
        # calculate the axis index
        ax_0 = i // 2
        ax_1 = i % 2

        # axies[ax_0,ax_1].legend(loc=2)

        print(ax_0, ax_1)

        axies[ax_0, ax_1].plot(sample[i]['Davg_shale']/10,sample[i]['Vp_dlogD_shale'],'bo-', markersize=8,
                               linewidth=2, alpha=1, label='Shale')
        axies[ax_0, ax_1].plot(sample[i]['Davg_kerogen']/10,sample[i]['Vp_dlogD_kerogen'],'y^-', markersize=8,
                               linewidth=2, alpha=1, label='Kerogen')
        #axies[ax_0, ax_1].legend(loc=2, )

        axies[ax_0, ax_1].plot([2,2],[0,30],'k--',linewidth = 2, alpha=0.5)
        axies[ax_0, ax_1].set_yticks([0, 0.005, 0.01, 0.015, 0.02])
        axies[ax_0, ax_1].set_xscale('log')
        axies[ax_0, ax_1].set_xticks([ 1.0, 10, 100])
        axies[ax_0,ax_1].set_xlim([1,250])
        axies[ax_0,ax_1].set_ylim([0,ylim_max[i]])
        if (ax_0 == 0) & (ax_1 == 0):
            axies[ax_0, ax_1].legend(loc=2,fancybox=True, framealpha=0.5)
        if i < 6:
            axies[ax_0, ax_1].set_xticks([])

        # plt.legend(handles=legends)
        if i == 0:
            axies[ax_0, ax_1].set_title('EF_' + names[i], fontsize=20, x=0.65, y=0.8)
        elif i < 3:
            axies[ax_0, ax_1].set_title('EF_' + names[i], fontsize=20, x=0.35, y=0.8)
        else:
            axies[ax_0, ax_1].set_title('NRM_' + names[i], fontsize=20, x=0.35, y=0.8)

        #axies[ax_0, ax_1].grid()
        #axies[ax_0, ax_1].set_ylim(ymin=-1)
        #if i == 4 or i== 3 or i ==2 :
            #axies[ax_0, ax_1].set_yticks([0, 4, 8, 12, 16])
        #elif i == 5 :
        #    axies[ax_0, ax_1].set_yticks([0, 2, 8, 12, 14])
    fig.text(0.5, 0.04, 'Pore width (nm)', ha='center')
    fig.text(0.05, 0.6, 'dV/dlog(w) (cm$^3$/g)', rotation=90)

    plt.tight_layout(pad=4, w_pad=0.5, h_pad=0.5)
    # fig.legend((line1,line2,line3),('HF','In','VF'),'upper left')

    plt.show()




#-------



def plot_porosity_bar():
    '''

    '''
    #direct = iso['direct']
#    figure = plt.figure
    #matplotlib.rcParams.update(params)
    filename = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/post_data.xlsx'
    df_porosity = pd.read_excel(filename,'N2_total')
    xticklabel = df_porosity['core_name']
    bulk_shale_porosity = df_porosity['In'][:8]
    kerogen_porosity = df_porosity['kerogen_porosity'][:8]# df_porosity['kerogen_porosity'][:8]
    fig,ax = plt.subplots(1,1)
    ind = np.arange(8)
    width =0.35
    #BET_select
    rects1 = ax.bar(ind, bulk_shale_porosity, width, color='b',alpha=1)
    rects2 = ax.bar(ind+width,kerogen_porosity, width, color='y',alpha=1)

    #ax.legend()
    ax.set_xlim([-0.2,8])
    plt.setp(ax, xticks=ind + width+0.02, xticklabels=xticklabel)
    ax.legend([rects1[0], rects2[0]], ['Shale', 'Kerogen',], fancybox=True, framealpha=0.5)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=20)
    #plt.xlabel('core names')
    plt.ylabel('Pore volume (cm$^3$/g)')
    plt.show()


def plot_fractal_dimension():
    filename = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/post_data.xlsx'
    df_fractal = pd.read_excel(filename, 'fractal_new')
    plt.plot(df_fractal['shale_D22'][3:8],df_fractal['kerogen_D22'][3:8],'go',label='NRM_kerogen')
    plt.plot(df_fractal['shale_D22'][3:8], df_fractal['clay_D22'][3:8], 'D',color='darkorchid',label='NRM_clay')
    plt.plot(df_fractal['shale_D22'][0:3], df_fractal['kerogen_D22'][0:3], 'ro', label='EF_kerogen')
    plt.plot([2.4,2.9],[2.4,2.9],'k--',linewidth=3)
    plt.xlim([2.62,2.72])
    plt.ylim([2.55,2.8])
    plt.xlabel('Bulk shale surface fractal dimension ')
    plt.ylabel('kerogen/clay surface fractal dimension ')

    plt.legend(loc=2,fancybox=True, framealpha=0.5)
    plt.show()


def plot_NRM_diameter():
    # plot the surface area vs pore volume
    filename = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/post_data.xlsx'
    df_fractal = pd.read_excel(filename, 'surface_vs_porosity')



    diameter_fracture_1 = 9.6
    diameter_fracture_2 = 9.6/2
    diameter_pore_1 = 9.6
    diameter_pore_2 = 4.5
    diameter_pore_3 = 4.5
    fig, axis = plt.subplots(1, 2)

    x1 = np.arange(-3, 20, 1)
    # x2 = np.arange(-3,0,0.1)
    y1 = x1 * diameter_fracture_1 / 2.0
    y2 = x1 * diameter_fracture_2 / 2.0
    y3 = x1 * diameter_pore_1 / 4.0
    y4 = x1 * diameter_pore_2 / 4.0
    y5 = x1 * diameter_pore_3 / 4.0

    axis[0].plot(df_fractal['in_BET(m2)'][3:8], df_fractal['in_total(m3)*10^3'][3:8], 'go')

    axis[0].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[0].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[0].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[0].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)

    axis[0].plot(x1, y1, 'r-')
    axis[0].plot(x1, y2, 'r-')
    axis[0].plot(x1, y3, 'k-')
    axis[0].plot(x1, y4, 'k-')
    axis[0].plot(x1, y5, 'k-')

    # axis[1].plot(x_line_range,x_line_range*diameter_pore_1/4.0,'g-')
    # axis[1].plot(x_line_range,x_line_range*diameter_pore_2/4.0,'g-')
    axis[0].set_xlim([0, 13])
    axis[0].set_ylim([0, 18])
    axis[0].set_xlabel('Surface area of intact sample (m$^2$/g)')
    axis[0].set_ylabel('Pore volume $ * 10^3$ (cm$^3$/g)')


    axis[1].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)

    axis[1].plot(x1, y1, 'r-')
    axis[1].plot(x1, y2, 'r-')
    axis[1].plot(x1, y3, 'k-')
    axis[1].plot(x1, y4, 'k-')
    axis[1].plot(x1, y5, 'k-')
    axis[1].plot(df_fractal['diff_BET(m2)'][3:8], df_fractal['diff_total_pore*10^3'][3:8], 'ro')
    axis[1].plot(df_fractal['diff_BET(m2)'][9], df_fractal['diff_total_pore*10^3'][9], 'r^')
    axis[1].plot([0, 0], [-3, 20], 'k--')
    axis[1].plot([-3, 20], [0, 0], 'k--')
    axis[1].set_xlim([-2, 4])
    axis[1].set_ylim([-3, 8])
    axis[1].set_xlabel('Surface area of change ( m$^2$/g)')

    # 3
    '''
    axis[2].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[2].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[2].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[2].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)

    axis[2].plot(x1, y1, 'r-')
    axis[2].plot(x1, y2, 'r-')
    axis[2].plot(x1, y3, 'k-')
    axis[2].plot(x1, y4, 'k-')
    axis[2].plot(x1, y5, 'k-')
    axis[2].plot(df_fractal['after_BET(m2)'][3:8], df_fractal['after_total(m3)*10^3'][3:8], 'ro')
    axis[2].plot(df_fractal['after_BET(m2)'][9], df_fractal['after_total(m3)*10^3'][9], 'r^')
    #axis[2].plot([0, 0], [-3, 20], 'k--')
    #axis[2].plot([-3, 20], [0, 0], 'k--')
    axis[2].set_xlim([0,18])
    axis[2].set_ylim([0,22])
    axis[2].set_xlabel('Surface area of change ( m$^2$/g)')
    '''


    plt.show()


def plot_EF_diameter():
    # plot the surface area vs pore volume
    filename = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/post_data.xlsx'
    df_fractal = pd.read_excel(filename, 'surface_vs_porosity')

    diameter_fracture_1 = 13.5
    diameter_fracture_2 = 13.5/2
    diameter_pore_1 = 13.5
    diameter_pore_2 = 12
    diameter_pore_3 = 12
    fig, axis = plt.subplots(1, 2)

    x1 = np.arange(-3, 30, 1)
    # x2 = np.arange(-3,0,0.1)
    y1 = x1 * diameter_fracture_1 / 2.0
    y2 = x1 * diameter_fracture_2 / 2.0
    y3 = x1 * diameter_pore_1 / 4.0
    y4 = x1 * diameter_pore_2 / 4.0
    y5 = x1 * diameter_pore_3 / 4.0

    axis[0].plot(df_fractal['in_BET(m2)'][0:3], df_fractal['in_total(m3)*10^3'][0:3], 'go')

    axis[0].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[0].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
    #axis[0].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    #axis[0].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[0].fill_between(x1, y3, y5, where=y5 >= y4, facecolor='green', alpha=0.3, interpolate=True)

    axis[0].plot(x1, y1, 'r-')
    axis[0].plot(x1, y2, 'r-')
    axis[0].plot(x1, y3, 'k-')
    axis[0].plot(x1, y4, 'k-')
    axis[0].plot(x1, y5, 'k-')

    # axis[1].plot(x_line_range,x_line_range*diameter_pore_1/4.0,'g-')
    # axis[1].plot(x_line_range,x_line_range*diameter_pore_2/4.0,'g-')
    axis[0].set_xlim([0, 9])
    axis[0].set_ylim([0, 25])
    axis[0].set_xlabel('Specific surface area (m$^2$/g)')
    axis[0].set_ylabel('Pore volume $ * 10^3$ (cm$^3$/g)')

    axis[1].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
    #axis[1].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    #axis[1].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y3, y5, where=y5 >= y4, facecolor='green', alpha=0.3, interpolate=True)
    axis[1].fill_between(x1, y3, y5, where=y5 < y4, facecolor='green', alpha=0.3, interpolate=True)

    axis[1].plot(x1, y1, 'r-')
    axis[1].plot(x1, y2, 'r-')
    axis[1].plot(x1, y3, 'k-')
    axis[1].plot(x1, y4, 'k-')
    axis[1].plot(x1, y5, 'k-')
    axis[1].plot(df_fractal['diff_BET(m2)'][0:3], df_fractal['diff_total_pore*10^3'][0:3], 'ro')
    axis[1].plot(df_fractal['diff_BET(m2)'][8], df_fractal['diff_total_pore*10^3'][8], 'r^')
    axis[1].plot([0, 0], [-3, 20], 'k--')
    axis[1].plot([-3, 20], [0, 0], 'k--')
    axis[1].set_xlim([-0.5, 1.5])
    axis[1].set_ylim([-1, 3])
    axis[1].set_xlabel('Specific surface area( m$^2$/g)')

    # 3
#    axis[2].fill_between(x1, y1, y2, where=y1 >= y2, facecolor='red', alpha=0.3, interpolate=True)
#    axis[2].fill_between(x1, y1, y2, where=y1 <= y2, facecolor='red', alpha=0.3, interpolate=True)
#    axis[2].fill_between(x1, y3, y4, where=y3 >= y4, facecolor='green', alpha=0.3, interpolate=True)
#    axis[2].fill_between(x1, y3, y4, where=y3 <= y4, facecolor='green', alpha=0.3, interpolate=True)
    '''
    axis[2].plot(x1, y1, 'r-')
    axis[2].plot(x1, y2, 'r-')
    axis[2].plot(x1, y3, 'k-')
    axis[2].plot(x1, y4, 'k-')
    axis[2].plot(x1, y5, 'k-')
    axis[2].plot(df_fractal['after_BET(m2)'][0:3], df_fractal['after_total(m3)*10^3'][0:3], 'ro')
    axis[2].plot(df_fractal['after_BET(m2)'][8], df_fractal['after_total(m3)*10^3'][8], 'r^')
    #axis[2].plot([0, 0], [-3, 20], 'k--')
    #axis[2].plot([-3, 20], [0, 0], 'k--')
    axis[2].set_xlim([0,9])
    axis[2].set_ylim([0,24])
    axis[2].set_xlabel('Surface area of change ( m$^2$/g)')
    '''
    plt.show()

####---main function----------------------------------------

params = {
   'axes.labelsize': 20,
   'font.size': 25,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [14, 18],
   }
#matplotlib.rcParams.update(params)
#----- plot
#plot_isotherm_per_core()
#get_fractal_dimension(i=1)


#----------------
params = {
   'axes.labelsize': 20,
   'font.size': 25,
   'legend.fontsize': 15,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
   'figure.figsize': [14, 18],
   }
matplotlib.rcParams.update(params)
#plot_psd_per_core()
# plot the porosity bar char
#


#---- plot fractal dimension
params = {
   'axes.labelsize': 35,
   'font.size': 30,
   'legend.fontsize': 32,
   'xtick.labelsize': 33,
   'ytick.labelsize': 33,
    'lines.markersize': 20,
   'text.usetex': False,
   'figure.figsize': [16, 14],
   }

#plot_fractal_dimension()
params = {
   'axes.labelsize': 20,
   'font.size': 25,
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'text.usetex': False,
    'lines.markersize': 10,
   'text.usetex': False,
   'figure.figsize': [16, 7.5],
   }
matplotlib.rcParams.update(params)
#plot_porosity_bar()

#plot_NRM_diameter()
plot_EF_diameter()





