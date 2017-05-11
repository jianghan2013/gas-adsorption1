from BJH_function import BJH_calculation
from visualization import N2_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
direct = 'C:/Users/hj5446/Dropbox/adsorption_toolbox/from_git/gas-adsorption1/Data_pyrolysis/'
file_name = direct+'N2_2_93.xlsx'
sheetnames =['ads_2_93_kerogen' ,'ads_2_93_110c','ads_2_93_250c','ads_2_93_450c','ads_2_93_650c']
for sheetname  in sheetnames:
    df_ads = pd.read_excel(file_name,sheetname=sheetname)
    p_ads,q_ads = df_ads['p'], df_ads['q']
    psd = BJH_calculation.BJH_method(p_ads, q_ads, use_pressure=True)
    psd.do_BJH()
    fractal_1, fractal_2 = N2_plot.get_fractal_number(p_ads, q_ads, False)
    print('---'+sheetname)
    print('total porosity  ',psd.vpore_total)
    print('fractal number 1  ',fractal_1['D12'],'\t',fractal_1['mabs_error_1'])
    print('fractal number 2  ',fractal_2['D22'],'\t',fractal_2['mabs_error_2'])
    #print(fractal_1['D12'],fractal_1,fractal_2['D22'])
    #['Davgfa'] = psd.Davg
    #iso[core_name][sample_name]['Vp'] = psd.Vp
    #plt.figure()


    fig,ax = plt.subplots()
    plt.plot(psd.Davg[1:],psd.Vp_dlogD[1:])
    ax.set_xscale('log')
    #plt.show()
    df_psd = pd.DataFrame({'Davg':psd.Davg[1:],'Vp_dlogD':psd.Vp_dlogD[1:]})
    df_psd.to_csv(direct+'psd_'+sheetname+'.csv',index=False)

