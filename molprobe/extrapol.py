import numpy as np


def func_parab( x, ln_a, b, c):
    return ln_a + b *(1./x - 1./c)**2

def func_high_temp(x, ln_tau0, ene_0):
    #x is the inverse of temperature
    return ln_tau0 + ene_0 * x