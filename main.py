

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn



def I1_function(t, eta, QCC, w0, tauc):

    A = 2/5 * np.pi**2 * ( 1 + eta**2/3 ) * QCC**2
    
    j1 = tauc / ( 1 + 1**2 * w0**2 * tauc**2 )
    j2 = tauc / ( 1 + 2**2 * w0**2 * tauc**2 )

    r1 = A * j1
    r2 = A * j2
    
    I = 1/5 * np.exp( -r1 * t) + 4/5 * np.exp( -r2 * t)

    return I

def I2_function():
    A = 2/5 * np.pi**2 * ( 1 + eta**2/3 ) * QCC**2



def fit_I1():
    df = pd.read_csv('I1.csv', sep=',', header=0)

    x = df['t'].values
    y = df['I'].values
    # fit curve
    popt, pcov = scipy.optimize.curve_fit(I1_function, x, y, p0=[1, 1, 1, 1])


    print('Fitted params [eta, QCC, w0, tauc]:\n', popt)
    
    print('Params covariance:\n', pcov)

    print('RMSF Error: ', np.sqrt(np.mean((I1_function(x, *popt) - y)**2)))

fit_I1()