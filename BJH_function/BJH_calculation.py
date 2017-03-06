import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

def get_gas_constant(gas_type='N2'):
    const = dict()
    # liquid molar volume [cm^3/mol]
    if gas_type == 'N2':
        const['A'] = 9.53
        const['Vmol'] = 34.67
    elif gas_type == 'Ar':
        const['A'] = 10.44
        const['Vmol'] = 22.56
    else:
        print('wrong gas type')
        const=-1
    return const

def insert_zero(a):
    return np.insert(a,0,0)

def kelvin_radius(p_rel,const):
    # Rc is in [A]
    # np.log is e base
    return -const['A'] / np.log(p_rel) 

def radius_to_pressure(Rc,const):
    #Rc is in [A]
    return np.exp(-const['A']/Rc)

def thickness_Harkins_Jura(p_rel):
    # t is in [A]
    return   (13.99 / (0.034 - np.log10(p_rel)))**0.5


def get_CSA_a(del_tw,Davg,LP,k,istep,n_step):
    # if it is the first step, no previous pore created
    if k==0 and istep < n_step: 
        Vd_istep = 0
    # if it is last step, no new pore will be created
    elif istep == n_step: 
        Vd_istep = 9999
    # calculate Vd 
    else: 
        #print('determine Vd >> 3 has old pore')
        Vd_istep =0
        CSA_a=np.zeros(k)
        CSA_a = insert_zero(CSA_a)
        for j in range(1,k+1):
            #CSA_a[j] = np.pi*((Rc[j]+ del_tw)**2-Rc[j]**2) *10**(-16)
            CSA_a[j] = np.pi*((Davg[j]/2.0+ del_tw)**2-(Davg[j]/2.0)**2) *10**(-16) # this one works better
            Vd_istep += LP[j]*CSA_a[j]
    return Vd_istep

#---------------- main function

def BJH(p,Q,gas_type='N2'):
    '''
    '''
    gas_const = get_gas_constant(gas_type)
    
    # insert the pressure=0, adsorption = 0 point 
    p = insert_zero(p)
    Q = insert_zero(Q)
    # make the isotherm in reverse order
    p_reverse = p[::-1]
    Q_reverse = Q[::-1]
    p_rels = np.zeros(len(p_reverse))
    q_ads  = np.zeros(len(p_reverse))
    p_rels[:] = p_reverse
    q_ads[:] = Q_reverse
    #print('old p_rels',p_rels,q_ads)
    p_rels,q_ads

    #
    VL = q_ads*gas_const['Vmol'] / 22414.0
    n_point = len(p_rels)
    n_step = n_point-1
    Vd = np.zeros(n_step)
    Vc = np.zeros(n_step)
    dV_desorp= np.zeros(n_step)
    status = np.zeros(n_step)
    tw = np.zeros(n_point)
    #print('old tw/status',n_point,n_step,tw,status)

    # not using first index
    p_rels, q_ads, tw, =insert_zero(p_rels), insert_zero(q_ads), insert_zero(tw)
    VL,Vd, Vc, dV_desorp, status = insert_zero(VL),insert_zero(Vd),insert_zero(Vc),insert_zero(dV_desorp),insert_zero(status)
    #print('new p_rels,q_ads',p_rels,q_ads)
    #print('VL',VL)
    #print('new tw/status',tw,status)

    # define other vector 
    Rc, Davg, Pavg= np.zeros(len(Vd)), np.zeros(len(Vd)), np.zeros(len(Vd))
    tw_avg, CSA_c, LP = np.zeros(len(Vd)), np.zeros(len(Vd)), np.zeros(len(Vd))
    #print('tw_avg',tw_avg)

    Rc[1]  = kelvin_radius(p_rels[1],gas_const)
    tw[1] = thickness_Harkins_Jura(p_rels[1])
    #print('Rc[1]/tw[1]',Rc[1],tw[1])

    k=0


    for istep in range(1,n_step+1):
        print('\nistep/nstep',istep,n_step)
        status[istep]= 0 
        #print(status)
        if istep == n_step:
            tw[istep+1]=0
        else:
            tw[istep+1] = thickness_Harkins_Jura(p_rels[istep+1])

        # a) determine Vd 
        del_tw = tw[istep] - tw[istep+1]
        #print('del_tw',del_tw)
        #print('Vd',Vd)
        Vd[istep] = get_CSA_a(del_tw,Davg,LP,k,istep,n_step)
        #print('Vd vs Vd_test',Vd[istep],Vd_test[istep])

        # b) check Vd with true desorption
        dV_desorp[istep] = VL[istep] - VL[istep+1]
        #print('dV_desorp',dV_desorp[istep])
        if Vd[istep] >= dV_desorp[istep]: # case 1: Vd too large

            status[istep] = 1
            #print('too large check case ',status[istep])
            #print('too large dV_desorp ',dV_desorp[istep])
            SAW = 0
            for j in range(1,k+1):
                SAW += np.pi*LP[j]*Davg[j] * 10**(-8)
            del_tw = dV_desorp[istep]/SAW  * 10**(8) # simplified version
            #print('SAW,new del_tw',SAW,del_tw)
        else:
            status[istep] = 2 # case 2: normal case
            #print('normal check case ',status[istep])
            Vc[istep] = dV_desorp[istep]- Vd[istep]
            #print('dV_desorp,Vc',dV_desorp[istep],Vc[istep])
            k += 1
            #print('n_pore',k)
            Rc[k+1]  = kelvin_radius(p_rels[k+1],gas_const)
            Davg[k] = 2* (Rc[k]+Rc[k+1]) *Rc[k]*Rc[k+1] / (Rc[k]**2+Rc[k+1]**2)
            Pavg[k] = np.exp(-2*gas_const['A'] / Davg[k])
            tw_avg[k] = thickness_Harkins_Jura(Pavg[k])
            del_td = tw_avg[k] - tw[istep+1]
            CSA_c[k] = np.pi*(Davg[k]/2.0+del_td)**2 *10**(-16)
            LP[k] = Vc[istep]/CSA_c[k]
            #print('Rc',Rc[k],Rc[k+1])
            #print('Vc,Davg,Pavg',Vc[istep],Davg[k],Pavg[k],tw_avg[k],CSA_c[k],LP[k])

        # c) updated pore diameter to the end pressure
        if status[istep]==2: # case 2 update current new pore diameter
            #print('updated new pore')
            Davg[k] += 2*del_td
        # no matter new pore created or not, updated previous diameter
        for j in range(1,k):
            #print('updated old pore',1,k-1)
            Davg[j] += 2*del_tw
        for j in range(1,k+1):
            Rc[j] += del_tw
        #print('Davg,Rc,LP',Davg,Rc,LP)

        # for test
        #temp1 = Davg.dot(LP)
        #Vp = np.pi*LP*(Davg/2.0)**2 *10**(-16)
        #Vp_cum = sum(Vp)
        #desorp_cum = sum(dV_desorp)
        #print('sum of Davg*LP*PI',temp1)
        #print('Vp_cum,total_desorp',Vp_cum,desorp_cum)

    #print(Davg)
    Dp  = 2*Rc
    return Davg,LP,Dp,dV_desorp,k

def result_psd(Davg,LP,Dp,k):
    Vp = np.pi*LP*(Davg/2.0)**2 *10**(-16) # return Vp vector[cm^3/g]
    Vp_ccum = np.add.accumulate(Vp)
    Vp_dlogD = np.zeros(len(Vp))
    for i in range(1,k+1):
        Vp_dlogD[i] = Vp[i]/ np.log10(Dp[i]/Dp[i+1])
    return Vp,Vp_ccum,Vp_dlogD
