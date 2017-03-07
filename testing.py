import unittest
from BJH_function import BJH_calculation
import numpy as np
import numpy.testing as npt

class testing(unittest.TestCase):

    def test_get_gas_constant(self):

        N2 = BJH_calculation.get_gas_constant()
        Ar = BJH_calculation.get_gas_constant('Ar')
        npt.assert_almost_equal(N2['A'],9.53,decimal=3)
        npt.assert_almost_equal(N2['Vmol'], 34.67, decimal=3)
        npt.assert_almost_equal(Ar['A'], 10.44)
        npt.assert_almost_equal(Ar['Vmol'], 28)

    def test_insert_zero(self):
        b = BJH_calculation.insert_zero([1,2])
        npt.assert_array_almost_equal(b,[0,1,2])

    def test_thickness_Harkins_Jura(self):
        p_rel = 0.499167054965542
        t = 6.45502909433275
        npt.assert_almost_equal(BJH_calculation.thickness_Harkins_Jura(p_rel),t,decimal=6)

    def test_pressure_and_radius(self):
        const_N2 = BJH_calculation.get_gas_constant('N2')
        const_Ar = BJH_calculation.get_gas_constant('Ar')
        npt.assert_almost_equal(BJH_calculation.kelvin_radius(0.5,const_N2),13.74888374,decimal=7)
        npt.assert_almost_equal(BJH_calculation.kelvin_radius(0.5,const_Ar), 15.06173623, decimal=7)
        npt.assert_almost_equal(BJH_calculation.radius_to_pressure(13.74888374, const_N2), 0.5, decimal=7)
        npt.assert_almost_equal(BJH_calculation.radius_to_pressure(15.06173623, const_Ar), 0.5, decimal=7)

    def test_get_CSA_a(self):
        # case 1
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(99,99,99,0,5,10),0)
        # case 2
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(99,99,99,99,9,9),9999)
        # case 3
        npt.assert_almost_equal(BJH_calculation.get_CSA_a(1, np.array([0,1,2]), np.array([0,1,3]), 2, 3, 4)*10**(16),34.55751919,decimal=7)

    def test_BJH_main_function(self):
        #3_14 n2
        from BJH_function import test_isotherm
        p_rels,q,my_volume = test_isotherm.shale_3_14()
        Davg,LP,Dp,dV_desorp,k = BJH_calculation.BJH(p_rels, q, 'N2')
        Vp,Vp_ccum,Vp_dlogD = BJH_calculation.result_psd(Davg,LP,Dp,k)
        npt.assert_array_almost_equal(Vp,my_volume,decimal=8)


        #assert np.assert_almost_equal
if __name__ == '__main__':
    unittest.main()