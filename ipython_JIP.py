
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


import matplotlib
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 20,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
   }
matplotlib.rcParams.update(params)


def plot_mineral_bar(df,keys =['Calcite','Illite/Mica','Mx I/S*','quartz','Plagioclase','Pyrite'],save =False):
    import operator

    #file_prechar ='C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_N2_triaxial/precharacterization_data.xlsx'
    #df_xrd = pd.read_excel(file_prechar,sheetname='xrd')

    fig,axis = plt.subplots(1,1,figsize=(14,10))

    width =0.8
    ind = np.arange(5)
        #print(data_micro,data_meso)
    #
    #keys = ['Calcite']
    #keys = ['Calcite','clay_total']
    rects = []
    bottom = tuple([0]*5)
    print(bottom)
    color_names = [ 'orange', 'crimson', 'darkorchid', 'g', 'b', 'black','sienna', 'grey']
    for i,key in enumerate(keys):
        print(i)
        if i > 0:
            a = tuple((df[keys[i - 1]][0:5]))
            bottom = tuple(map(operator.add, a, bottom))
        rect = axis.bar(ind,df[key][0:5],width,bottom=bottom,color=color_names[i])
        rects.append(rect)
    # add 'other' as the last mineral
    a = tuple((df[keys[-1]][0:5]))
    #bottom = tuple(map(operator.add, a, bottom))
    #rect = axis.bar(ind,100 - np.array(bottom),width,bottom=bottom,color=color_names[i+1])
    #rects.append(rect)
    plt.ylim([0,100])
        #if key == 'other':
            #rect = axis.bar(ind, df_xrd['quartz'][0:8], width, bottom=bottom, color=colors[i])
            #b  = np.array([100]*8) - np.array(bottom)
            #print(0)
            #rect = axis.bar(ind, np.array([100]*8)--, width, bottom=bottom, color=colors[i])

        #print('a',a)

        #print('bottom',bottom)

        #axies[ax_0,ax_1].set_ylim([0, 0.04])
        #axies[ax_0,ax_1].set_title(core_name)

        #plt.title('pore_'+core_name)
        #plt.xticks(ind+width/2,xticklabel)
    plt.setp(axis,xticks=ind+width/2,xticklabels=df['temp'][0:5])
        #axies[ax_0,ax_1].set_xticks(xticklabel)
    keys.append('other')
    #axis.legend(rects,keys,fancybox=True, )
    box = axis.get_position()
    axis.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.ylabel('weight percentage (%)')
    axis.legend(rects,keys,loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=8,framealpha=1)
    #plt.legend()
        #figure = plt.figure(core_name)
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axies[0, :]], visible=False)
    #for k in range(1,4):
        #plt.setp([a.get_yticklabels() for a in axies[:, k]], visible=False)
    #plt.tight_layout()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

file_location = 'F:/1Research/NMR_Inverse Code/Measurement/results/'
filename = file_location+'pyrolysis_NMR_results.xlsx'
df = pd.read_excel(filename,sheetname='relative porosity_2_93')
keys=['z1','z2','z3','z4','z5','z6','z7']
#print(df)
#print(df['z3'])
plot_mineral_bar(df,keys)
