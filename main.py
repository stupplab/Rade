

import numpy as np
import pandas as pd
import scipy.optimize


def I1_function(t, eta, QCC, w0, tauc):
    A = 2/5 * np.pi**2 * ( 1 + eta**2/3 ) * QCC**2
    j1 = tauc / ( 1 + 1**2 * w0**2 * tauc**2 )
    j2 = tauc / ( 1 + 2**2 * w0**2 * tauc**2 )
    r1 = A * j1
    r2 = A * j2
    I = 1/5 * np.exp( -r1 * t) + 4/5 * np.exp( -r2 * t)
    return I


def I2_function(t, eta, QCC, w0, tauc):
    A = 2/5 * np.pi**2 * ( 1 + eta**2/3 ) * QCC**2
    j0 = tauc
    j1 = tauc / ( 1 + 1**2 * w0**2 * tauc**2 )
    j2 = tauc / ( 1 + 2**2 * w0**2 * tauc**2 )
    s1 = 1/2 * A * (j0 + j1)
    s2 = 1/2 * A * (j1 + j2)
    I = 3/5 * np.exp(-s1*t) + 2/5 * np.exp(-s2*t)
    return I


def fit_I1():
    df = pd.read_csv('I1.csv', sep=',', header=0)
    x = df['t'].values
    y = df['I'].values
    p0 = [1, 1, 1, 1]
    popt, pcov = scipy.optimize.curve_fit(I1_function, x, y, p0=p0)
    print('Fitted params [eta, QCC, w0, tauc]:\n', popt)
    print('Params covariance:\n', pcov)
    print('RMSF Error: ', np.sqrt(np.mean((I1_function(x, *popt) - y)**2)))

fit_I1()


def fit_I2():
    df = pd.read_csv('I2.csv', sep=',', header=0)
    x = df['t'].values
    y = df['I'].values
    p0 = [1, 1, 1, 1]
    popt, pcov = scipy.optimize.curve_fit(I2_function, x, y, p0=p0)
    print('Fitted params [eta, QCC, w0, tauc]:\n', popt)
    print('Params covariance:\n', pcov)
    print('RMSF Error: ', np.sqrt(np.mean((I1_function(x, *popt) - y)**2)))

fit_I2()



















