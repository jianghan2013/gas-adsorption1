import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
def get_gas_constant(gas_type='N2'):
    const = dict()
    # liquid molar volume [cm^3/mol]
    if gas_type == 'N2':
        const['A'] = 9.53
        const['Vmol'] = 34.67
    elif gas_type == 'Ar':
        const['A'] = 10.44
        const['Vmol'] = 28#22.56
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

def restrict_isotherm( P, Q, Pmin, Pmax ):
    """Restrict the isotherm Q, P to pressures between min and max.

    Q: Quanity adsorbed
    P: Pressure (relative or absolute)
    Pmin: minimum pressure
    Pmax: maximum pressure

    Returns:  Qads, P restricted to the specified range
    """

    index = (P >= Pmin) & (P <= Pmax)
    #b = np.logical_and( P >= Pmin, P <= Pmax)
    return P[index], Q[index]

def get_spline(p_rels,Q,order=2):
    s = InterpolatedUnivariateSpline(p_rels,Q,k=order)
    return s

def use_my_pressure_points(p_raw,q_raw,gas_type):
    const = get_gas_constant(gas_type)
    # predefine the radius
    radius_s = np.array([8,9,10,11,12,13,14,16,19,27,42,75,140,279,360,460,828,1600])
    # convert radius to pressure
    p_s = radius_to_pressure(radius_s,const)
    func_spline = get_spline(p_raw,q_raw,2)
    # get Q_s from spline function
    q_s = func_spline(p_s)
    return p_s,q_s

def get_porosity(p_raw,q_raw,gas_type='N2'):
    '''
    calculate the porosity of total, meso and micro
    the total porososity is corresponding to total volume of radius <= 160nm
    :param p: relative pressure points
    :param q: adsorption amount
    :param gas_type: type of gas
    :return: v_pore_total,v_pore_micro,v_pore_meso
    '''
    const = get_gas_constant(gas_type)
    r_micro = 8 # lagest micro pore radius [am]
    r_meso = 1600 # lagest meso/macro pore radius [am]
    p_micro = radius_to_pressure(r_micro, const)
    p_meso = radius_to_pressure(r_meso, const)
    func_spline = get_spline(p_raw, q_raw, 2)
    q_micro = func_spline(p_micro)
    q_meso = func_spline(p_meso)

    #p_res, q_res = restrict_isotherm(p, q, Pmin=0.3, Pmax=0.999)
    #p0,q0 = p_res[0],q_res[0]
    #p1,q1 = p_res[-1],q_res[-1]
    #print(p0,q0,p1,q1)
    #gas_const = get_gas_constant(gas_type)
    vpore_total = q_meso*const['Vmol'] / 22414.0
    vpore_micro = q_micro*const['Vmol'] / 22414.0
    vpore_meso = vpore_total - vpore_micro
    return vpore_total,vpore_micro,vpore_meso

#---------------- main calculation function

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
        #print('\nistep/nstep',istep,n_step)
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

#---------------- main function-----------------
def BJH_main(p,Q,pmin=0.30,pmax=0.999,use_pressure=True,gas_type='N2'):
    '''
    currently  not in use
    :argument
    :param p:
    :param Q:
    :param pmin:
    :param pmax:
    :param gas_type:
    :return:
    '''
    if use_pressure:
        p,Q = use_my_pressure_points(p, Q, gas_type)
    p_res,Q_res = restrict_isotherm( p, Q, pmin, pmax )
    Davg, LP, Dp, dV_desorp, k = BJH(p_res,Q_res,gas_type)
    Vp, Vp_ccum, Vp_dlogD = result_psd(Davg,LP,Dp,k)
    return Davg,Vp,Vp_ccum,Vp_dlogD

#------------------------- class ---------------------------------------

class BJH_method():
    def __init__(self,p,q,pmin=0.30,pmax=0.999,use_pressure=True,gas_type='N2'):
        self.p_raw = p
        self.q_raw = q
        self.p = p
        self.q = q
        self.gas_type = gas_type
        self.use_pressure = use_pressure
        if use_pressure:
            self.p, self.q = use_my_pressure_points(self.p_raw, self.q_raw, self.gas_type)
        else:
            self.p, self.q = self.p_raw,self.q_raw
        self.p_res, self.q_res = restrict_isotherm(self.p, self.q, pmin, pmax)
            

    def plot_isotherm(self):
        figure = plt.figure()
        legend = []
        legend_raw, = plt.plot(self.p_raw,self.q_raw,'ko-',label='raw iso')
        legend.append(legend_raw)
        if self.use_pressure:
            legend_fix, = plt.plot(self.p, self.q, 'r.',label='fixed iso')
            legend.append(legend_fix)
        plt.legend(handles=legend,loc=2)
        #plt.grid()


    def do_BJH(self):
        #if self.use_pressure: #only when we using fixed pressure point
        self.vpore_total, self.vpore_micro, self.vpore_meso = get_porosity(self.p_raw,self.q_raw,gas_type=self.gas_type)
        self.Davg, self.LP, self.Dp, self.dV_desorp, self.k = BJH(self.p_res, self.q_res, self.gas_type)
        self.Vp, self.Vp_ccum, self.Vp_dlogD = result_psd(self.Davg, self.LP, self.Dp, self.k)
		

    def plot_BJH_psd(self,plot_type = 'incremental'):
        figure = plt.figure()
        legend = []
        if plot_type ==   'incremental':
            plt.title('PSD by incremental pore volume')
            legend_incre, = plt.semilogx(self.Davg[1:], self.Vp[1:], 'go-', label='incremental')
            legend.append(legend_incre)
        plt.legend(handles=legend)
        #plt.grid()

