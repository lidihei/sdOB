import numpy as np

def weighted_average(x, err, w=None):
    '''
    calculate the weigthed arerage, weight, w_i = \frac{1/err_i^2}{sum{1\err_i^2}}
    parameters:
    -------------------------------
    x: [1D array]
    err: [1D array] 
    w: [1D array] the weight, if w is None: w = 1/err**2
    returns:
    ------------------------------
    x_average
    err_average
    '''
    if w is None: w = 1/err**2
    wsum = np.nansum(w)
    wi = w/wsum
    x_average = np.nansum(wi*x)
    err_average = np.sqrt(np.nansum((wi*err)**2))
    return x_average, err_average


def cubicequation_solution(a, b, c, d):
    ''' calculate the solutions of cubic equation: a*x^3 + b*x^2 +c*x +d = 0
    #https://baike.baidu.com/item/%E8%A7%A3%E4%B8%80%E5%85%83%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B/9386910
    returns:
    ------------
    x1, x2, x3
    '''
    A = b**2 - 3*a*c
    B = b*c - 9*a*d
    C = c**2 - 3*b*d
    Delta = B**2 -4*A*C
    if ((A==B) and (A==0)):
        x1 = -b/3/a
        x2 = x1
        x3= x1
    if Delta > 0:
        AB = A*b
        ABC = B**2-4*A*C
        Y1 = AB + 1.5*a*(-B + np.sqrt(ABC))
        Y2 = AB + 1.5*a*(-B - np.sqrt(ABC))
        YY1 = np.sign(Y1)*np.abs(Y1)**(1/3)
        YY2 = np.sign(Y1)*np.abs(Y2)**(1/3)
        YYp = YY1 + YY2
        YYm = YY1-YY2
        a3 = 3*a
        x1 = (-b-YYp)/a3
        x2 = (-b+0.5*YYp + np.sqrt(3)/2*YYm*1*1j)/a3
        x3 = (-b+0.5*YYp - np.sqrt(3)/2*YYm*1*1j)/a3
    if Delta == 0:
        K = B/A
        x1 = -b/a +K
        x2 = -K/2
        x3 =2
    if Delta < 0:
        T = (2*A*b - 3*a*B)/2/np.sqrt(A**3)
        print(T)
        th = np.arccos(T)/3
        Ar = np.sqrt(A)
        a3 = 3*a
        x1 = (-b-2*Ar*np.cos(th))/a3
        x2 = (-b -Ar*(np.cos(th)+np.sqrt(3)*np.sin(th)))/a3
        x3 = (-b -Ar*(np.cos(th)-np.sqrt(3)*np.sin(th)))/a3
    return x1, x2, x3

