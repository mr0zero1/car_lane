from scipy.optimize import curve_fit
import numpy as np
def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

x=[1,2,3]
y=[4,5,6]
A,B = curve_fit(f, x, y)[0] # your data x, y to fit

print(A,B)
y =np.array(x)*A + B
print(y)
