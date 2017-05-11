import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import unittest
import numpy.testing as npt

#---------------------------------------
def linear_fit(x_train,y_train,do_plot=True, fit_intercept = True):
    '''
    to do the linear fit for the
    :param x_train(1D array): input x data
    :param y_train(1D array): input y data
    :param do_plot(bollean): plot the results
    :param fit_intercept(bollean): if False, then intercept is 0
    :return:
    y_hat_train(1D array): the prediction of y based on linear model
    coeffs(dict()):
        coeffs['slope'] =  slope
        coeffs['intercept'] = intercept
        coeffs['R2'] = R2
        coeffs['standard_error_slope'] = standard_error_slope
    '''

    # reshape the input
    x_train = np.array(x_train).reshape(-1,1)
    y_train = np.array(y_train).reshape(-1,1)

    # initiate model  ; determine whether set intescept == 0 or not
    regression_model = linear_model.LinearRegression(fit_intercept= fit_intercept)
    regression_model.fit(x_train, y_train)
    slope = regression_model.coef_[0][0]
    intercept = regression_model.intercept_[0] if fit_intercept is True else 0

    # prediciton
    y_hat_train = regression_model.predict(x_train)

    #  y-y_hat
    error = y_hat_train - y_train
    sq_error = error **2
    residual_sum_sq = np.sum(sq_error)

    # y -y_bar
    error_y = y_train - np.average(y_train)
    total_sum_sq = np.sum(error_y**2)

    # R2
    R2  = 1 - residual_sum_sq / total_sum_sq

    # x - x_bar
    error_x = x_train - np.average(x_train)
    x_sum_sq = np.sum(error_x **2)

    # standard error
    standard_error_slope = np.sqrt( residual_sum_sq/(len(x_train)-2)/x_sum_sq)
    #print(r2)
    if do_plot:
        x_min = float(np.amin(x_train)) if fit_intercept is True else 0
        x_max = float(np.amax(x_train))
        x_test = np.array([x_min,x_max]).reshape(-1,1)
        y_hat_test = regression_model.predict(x_test)

        plt.plot(x_train,y_train,'o')
        plt.plot(x_test,y_hat_test,'r-')
        plt.show()


    coeffs = dict()
    coeffs['slope'] =  slope
    coeffs['intercept'] = intercept
    coeffs['R2'] = R2
    coeffs['standard_error_slope'] = standard_error_slope

    return y_hat_train,coeffs


#--------unittesting-------------------------------------------------------
class testing(unittest.TestCase):
    def testing_all(self):
        x_train = [1, 2, 4, 5, 3, 2]
        y_train = [4, 3, 2, 4, 1, 4]
        y_hat_train, coeffs = linear_fit(x_train, y_train, do_plot = False, fit_intercept=True)
        npt.assert_almost_equal(coeffs['slope'],-0.184615385 , decimal=8)
        npt.assert_almost_equal(coeffs['intercept'], 3.523076923, decimal=8)
        npt.assert_almost_equal(coeffs['R2'], 0.0461538461538, decimal=8)
        npt.assert_almost_equal(coeffs['standard_error_slope'], 0.419636359907, decimal=8)

if __name__ == '__main__':
    unittest.main()
