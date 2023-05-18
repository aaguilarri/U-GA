#  objective functions
import numpy as np
import random
import sys


def one_max(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    if modo == 'binary':
        return sum(x)
    else:
        suma = 0
        numo = int(x[0])
        signo = np.sign(numo)
        numo = abs(numo)
        print(f"numo is {numo}")
        while numo > 1:
            suma += numo % 2
            numo = int(numo/2)
            print(f"numo is {numo}")
        return signo * (suma + np.round(numo))


def zero_max(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    if modo != 'binary':
        print('This function does not supports real mode')
        return
    lx = len(x)
    return lx - sum(x)


def sq01(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    lx = len(x)
    ix = np.linspace(lx - 1, 0, lx)
    twos = 2 * np.ones(lx)
    ex = np.power(twos, ix)
    numba = x @ ex
    numba2 = (np.power(2., lx) - 1.) * numba - np.power(numba, 2)
    return numba2


def quad01(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    a = 5.
    b = 2.
    c = -3.
    lx = len(x)
    ix = np.linspace(lx - 1, 0, lx)
    twos = 2 * np.ones(lx)
    ex = np.power(twos, ix)
    nb = x @ ex
    # print('x = ', x)
    # print('nb = ', nb)
    # print('twos = ', twos)
    tn1 = np.power(2., lx) - 1
    # print('tn1 = ', tn1)
    xx = (5. * nb) / (3. * tn1) - 1.
    # print('xx = ', xx)
    # print('****************************************')
    y = a * np.power(xx, 3) + b * np.power(xx, 2) + c * xx
    return y


def lines01(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    lx = len(x)
    ix = np.linspace(lx - 1, 0, lx)
    twos = 2 * np.ones(lx)
    ex = np.power(twos, ix)
    nb = x @ ex
    tn1 = np.power(2., lx) - 1
    xx = 10. * nb / tn1
    if xx < 5.:
        return -xx / 5. + 1.
    else:
        return -2. * xx / 5 + 4.


def lines02(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    lx = len(x)
    ix = np.linspace(lx - 1, 0, lx)
    twos = 2 * np.ones(lx)
    ex = np.power(twos, ix)
    nb = x @ ex
    tn1 = np.power(2., lx) - 1
    xx = 10. * nb / tn1
    if xx < 5.:
        return xx / 5.
    else:
        return -2. * xx / 5 + 4.


def trap01(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    n = 10.
    n1 = n - 1
    lx = len(x)
    ix = np.linspace(lx - 1, 0, lx)
    twos = 2 * np.ones(lx)
    ex = np.power(twos, ix)
    nb = x @ ex
    # print('nb ', nb)
    tn1 = np.power(2., lx) - 1
    xx = n * nb / tn1
    # print('xx ', xx)
    if xx < n1:
        return -xx + n1
    else:
        return n * xx / (n - n1) - (n * n1) / (n - n1)

# from Goldberg, D. E., Sastry, K., & Ohsawa, Y. (2003).
# Discovering deep building blocks for competent genetic algorithms using
#  chance discovery via keygraphs. In Chance Discovery (pp. 276-301).
#  Springer, Berlin, Heidelberg.


def trap02(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    if modo != 'binary':
        print(f"Mode {modo} is not supported for this function")
        return
    u = sum(x)
    delta = 0.25
    k = len(x) - 1
    if u == k:
        return 1.
    else:
        return (1. - delta) * (1. - u / (k - 1.))


def mxn_problem(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    if modo != 'binary':
        print(f"Mode {modo} is not supported for this function")
        return
    m = my_m
    q = int(len(x)/m)
    xs = np.array_split(x, q)
    res = 0
    for xi in xs:
        res += trap02(xi)
    return res


def p_2x4p12(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    if modo != 'binary':
        print(f"Mode {modo} is not supported for this function")
        return
    x1 = x[:3]
    x2 = x[4:8]
    x3 = x[8:]
    return trap02(x1) + trap02(x2) + trap02(x3)

# A Literature Survey of Benchmark Functions For Global Optimization Problems
# and
# Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark Functions
# Repository [https://www.al-roomi.org/benchmarks/unconstrained]. Halifax,
# Nova Scotia, Canada: Dalhousie University, Electrical and Computer Engineering.


def beale(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -4.5
    ymax = 4.5
    z1 = 0
    z2 = 0
    q = 2
    zs = pre_process(x, [ymin, ymax], q, modo=modo)
    z1 = zs[0]
    z2 = zs[1]
    res = np.power(1.5 - z1 + z1*z2, 2) + np.power(2.25 - z1 + z1*z2*z2, 2) + \
        np.power(2.625 - z1 + z1*z2*z2*z2, 2)
    return -res  # - to make it maximization
    # x1 = 11010100 212 3
    # x2 = 10001101 141 0.5
    # y* = 0.0


def chung_reynolds(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    m = my_m
    ymin = -100
    ymax = 100
    chunks = []
    suma = 0
    q = int(len(x) / m)
    chunks = pre_process(x, [ymin, ymax], q, modo=modo)
    for chunk in chunks:
        val = chunk  # to_dec(chunk, ymin, ymax)
        suma += val * val
    return - (suma**2)


def dixon_price(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    m = my_m
    ymin = -10
    ymax = 10
    q = int(len(x) / m)
    nums = pre_process(x, [ymin, ymax], q, modo=modo)
    # nums = [1, 1/np.sqrt(2)]
    if len(nums) < 2:
        print('At least 2 chunks needed')
        return
    suma = (nums[0] - 1)**2
    for i in range(1, len(nums)):
        item = (i+1)*(2*(nums[i]**2) - nums[i-1])**2
        suma += item
    return -suma
    # x* = f(2((2^i -2)/(2^i))= 0 [1, 1/np.sqrt(2)]


def el_attar_vidyasagar_dutta(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -500.
    ymax = 500.
    x1 = 0
    x2 = 0
    q = int(len(xx)/2)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    [x1, x2] = [x[0], x[1]]
    # x1 = 3.4091868222
    # x2 = -2.1714330361
    # y = 1.712780354862198
    y = np.power(x1*x1 + x2 - 10, 2) + np.power(x1 + x2*x2
                                                - 7, 2) + np.power(x1*x1 + x2*x2*x2 - 1, 2)
    return -y
    # -x∗ = -f(3.4091868222, -2.1714330361) = -1.712780354862198


def matyas(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -10.
    ymax = 10.
    x1 = 0
    x2 = 0
    q = int(len(xx)/my_m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    [x1, x2] = [x[0], x[1]]
    y = 0.26*(x1**2 + x2**2) - 0.48*x1*x2
    return -y
    # -x* = -f(0,0) = 0


def powell_singular(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -4.
    ymax = 5.
    nums = []
    q = 0
    lq = 0
    q = int(len(x) / my_m)
    nums = pre_process(x, [ymin, ymax], q, modo=modo)
    lq = int(len(nums) / 4)
    if len(nums) % 4 != 0:
        print('Chunks must be multples of 4')
        return
    y = 0
    # nums = [3, -1, 0, 1, 3, -1, 0, 1]
    for i in range(0, lq):
        x4i = nums[4*i]
        x4i_1 = nums[1 + 4*i]
        x4i_2 = nums[2 + 4*i]
        x4i_3 = nums[3 + 4*i]
        # print(f"x0={x4i}, x1={x4i_1}, x2={x4i_2}, x3={x4i_3}")
        y += (x4i_3 + 10*x4i_2)**2 + 5*(x4i_1 - x4i)**2 + (x4i_2 - x4i_1)**4
        + 10*(x4i_3 - x4i)
    return -y
    # -f(x*) = -f(0,0,0,0 ... 0,0,0,0) = 0


def powell_singular_2(x, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -4.
    ymax = 5.
    m = my_m
    q = int(len(x) / m)
    nums = pre_process(x, [ymin, ymax], q, modo=modo)
    lq = len(nums) - 2
    y = 0
    for i in range(1, lq):
        xim1 = nums[i - 1]
        xi = nums[i]
        xip1 = nums[i + 1]
        xip2 = nums[i + 2]
        # print(f"({xim1}, {xi}, {xip1}, {xip2})")
        y += np.power(xim1 + 10*xi, 2) + np.power(5*(xip1 - xip2), 2) + \
            np.power(xi - 2*xip1, 4) + np.power(10*(xim1 - xip2), 4)
    return -y
    # -f(x*) = -f(0,0,0,0 ... 0,0,0,0) = 0


def powell_sum(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -1.
    ymax = 1.
    q = int(len(xx) / my_m)
    nums = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0
    ex = 1
    for nm in nums:
        # print(f"nm={nm}, ex={ex+1}")
        y += np.power(abs(nm), ex + 1)
        ex += 1
    return -y
    # -f(x*) = -f(0,0,0,0 ... 0,0,0,0) = 0


def rosenbrock(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -30.
    ymax = 30.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    res = 0
    for i in range(0, len(x) - 1):
        res += 100*np.power(x[i+1] - np.power(x[i], 2),
                            2) + np.power(x[i] - 1, 2)
    return -res


def rotated_ellipse(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -500.
    ymax = 500.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0
    y = 7*np.power(x[0], 2) - 6*np.sqrt(3)*x[0]*x[1] + 13*np.power(x[1], 2)
    return -y


def rotated_ellipse_2(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -500.
    ymax = 500.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    # print('x = ', x)
    y = 0
    y = np.power(x[0], 2) - x[0]*x[1] + np.power(x[1], 2)
    return -y


def rump(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -500.
    ymax = 500.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    # print('x = ', x)
    y = 0
    x1 = x[0]
    x2 = x[1]
    y = (333.75 - np.power(x1, 2))*np.power(x2, 6)
    z = 11*np.power(x1, 2) * np.power(x2, 2) - 121*np.power(x2, 4) - 2
    y += np.power(x1, 2) * z
    y += 5.5*np.power(x2, 8)
    if x1 != 0. and x2 != 0:
        y += x1/(2*x2)
    return -abs(y)


def scahffer_1(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0.5
    a = np.power(np.sin(np.power(np.power(x[0], 2) + np.power(x[1], 2), 2)), 2)
    a -= 0.5
    b = np.power(1 + 0.001*(np.power(x[0], 2) + np.power(x[1], 2)), 2)
    y += (a/b)
    return -y
    # fmin(X∗)=0
    # x∗i=0


def scahffer_2(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0.5
    a = np.power(np.sin(np.power(np.power(x[0], 2) - np.power(x[1], 2), 2)), 2)
    a -= 0.5
    b = np.power(1 + 0.001*(np.power(x[0], 2) + np.power(x[1], 2)), 2)
    y += (a/b)
    return -y
    # fmin(X∗)=0
    # x∗i=0


def scahffer_3(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0.5
    a = np.power(
        np.sin(np.cos(np.abs(np.power(x[0], 2) - np.power(x[1], 2)))), 2)
    a -= 0.5
    b = np.power(1 + 0.001*(np.power(x[0], 2) + np.power(x[1], 2)), 2)
    y += (a/b)
    return -y
    # Four global minimum are known: fmin(X∗)=0.001566854526004
    # x∗i≈(0,±1.253114962205510),(±1.253114962205510,0)


def scahffer_4(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    q = 2
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    # x=[0, 1.253131828792882]
    y = 0.5
    a = np.power(
        np.cos(np.sin(np.abs(np.power(x[0], 2) - np.power(x[1], 2)))), 2)
    a -= 0.5
    b = np.power(1 + 0.001*(np.power(x[0], 2) + np.power(x[1], 2)), 2)
    y += (a/b)
    return -y
    # Four global minimum are known: fmin(X∗)=0.292578632035980
    # x∗i≈(0,±1.253131828792882),(±1.253131828792882,0)


def sphere(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):  # Schummer-Steiglitz Func 1
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    # print('x = ', x)
    y = 0
    for chunk in x:
        y += np.power(chunk, 2)
    return -y
    # fmin(X∗)=0
    # x∗i≈(0, 0, ... , 0)


def schwefel_1_2(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    # print('x = ', x)
    y = 0
    for i in range(0, len(x)):
        z = 0
        for j in range(0, i):
            z += x[j]
        y += np.power(z, 2)
    return -y
# fmin(X∗)=0
# x∗i=0


def schwefel_2_21(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    z = []
    for i in x:
        z.append(abs(i))
    y = max(z)
    return -y
# x ∗ = f (0, · · · , 0),
# f (x ∗ ) = 0


def schwefel_2_22(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    x = [np.abs(chunk) for chunk in x]
    #print('x = ', x)
    y = sum(x) + np.prod(x)
    return -y
# x ∗ = f (0, · · · , 0),
# f (x ∗ ) = 0


def step_1(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    x = [np.floor(np.abs(chunk)) for chunk in x]
    # print('x = ', x)
    y = sum(x)
    return -y
# x ∗ = f (0, · · · , 0),
# f (x ∗ ) = 0


def step_2(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    x = [np.power(np.floor(chunk + 0.5), 2) for chunk in x]
    # print('x = ', x)
    y = sum(x)
    return -y
# x ∗ = f (0, · · · , 0),
# −0.5≤x∗i<0.5 → i.e., x∗i∈[−0.5,0.5)


def step_3(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -100.
    ymax = 100.
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    x = [np.floor(np.power(chunk, 2)) for chunk in x]
    y = sum(x)
    return -y
# Infinite number of m = my_minimum: fmin(X∗)=0
# −1<x∗i<1 → i.e., x∗i∈(−1,1)


def stepint(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = -5.12
    ymax = 5.12
    m = my_m
    q = int(len(xx) / m)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    x = [np.floor(chunk) for chunk in x]
    y = 25 + sum(x)
    return -y
# Infinite number of global minimum: fmin(X∗)=0
# −1<x∗i<1 → i.e., x∗i∈(−1,1)


def type_1_deceiptive(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = 0
    ymax = 1
    m = my_m
    q = int(len(xx) / m)
    if alphas == []:
        alphas = alpha*np.ones(q)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0
    for xi in x:
        y += gof(xi, alpha)
    y /= len(x)
    return np.power(y, beta)
# fmax(X∗)=1 with 2n−1 local maximum points
# x∗i=1


def type_2_deceiptive(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = 0
    ymax = 1
    m = my_m
    q = int(len(xx) / m)
    if alphas == []:
        alphas = alpha*np.ones(q)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0
    for xi, ai in zip(x, alphas):
        y += gof2(xi, alpha, ai)
    y /= len(x)
    return np.power(y, beta)
# It has two global maximum: fmax(X∗)=1 with 2n−1 local maximum
# x∗i={0,1}
# For n=1, there are two possibilities for f(x), because it is constructed by one g(x). While for n=2,
# there are four possibilities for f(X), because it depends on the summation of g1(x1) and g2(x2).


def type_3_deceiptive(xx, alpha=0.8, beta=1, my_m=8, alphas=[], modo='binary'):
    ymin = 0
    ymax = 1
    m = my_m
    q = int(len(xx) / m)
    if alphas == []:
        alphas = alpha*np.ones(q)
    x = pre_process(xx, [ymin, ymax], q, modo=modo)
    y = 0
    for xi, alpha in zip(x, alphas):
        y += gof3(xi, alpha)
    y /= q
    return np.power(y, beta)
# --------------------------------------------------------


def to_dec(b, ymin, ymax):
    lb = len(b)
    # print('b ', b)
    # print('lb ', lb)
    ix = np.linspace(lb - 1, 0, lb)
    twos = 2 * np.ones(lb)
    ex = np.power(twos, ix)
    # print(f"ex = {ex}")
    nb = b @ ex
    nbm = np.power(2, lb) - 1
    # print('ymax ', ymax)
    # print('ymin ', ymin)
    # print('nb ', nb)
    # print('nbm ', nbm)
    res = (ymax - ymin) * (nb / nbm) + ymin
    # print(f"res {res}")
    return res


def gof(x, alpha):
    if x >= 0 and x <= alpha:
        return alpha - x
    else:
        return (x - alpha)/(1 - alpha)


def gofb(x, alpha):
    if x >= 0 and x <= 1 - alpha:
        return (1 - alpha - x)/(1 - alpha)
    else:
        return x - 1 + alpha


def gof2(x, alpha, ai):
    if ai == 0:
        return gof(x, alpha)
    else:
        return gofb(x, alpha)


def gof3(x, alpha):
    if x >= 0 and x < 4*alpha/5:
        return -x/alpha + 4/5
    if x >= 4*alpha/5 and x < alpha:
        return 5*x/alpha - 4
    if x >= alpha and x < (1 + 4*alpha)/5:
        return 5*(x - alpha)/(alpha - 1) + 1
    return (x-1)/(1-alpha) + 4/5


def pre_process(xx, ys, q, modo='binary'):
    x = []
    [ymin, ymax] = [ys[0], ys[1]]
    if modo == 'binary':
        chunks = np.array_split(xx, q)
        #print(f"chunks = {chunks}, {ymin}, {ymax}")
        for chunk in chunks:
            x.append(to_dec(chunk, ymin, ymax))
    else:
        x = [min(max(chunk, ymin), ymax) for chunk in xx]
    return x
#  Function dictionary #######################################################


func_dict = dict()
func_dict['one_max'] = one_max
func_dict['zero_max'] = zero_max
func_dict['sq01'] = sq01
func_dict['quad01'] = quad01
func_dict['lines01'] = lines01
func_dict['lines02'] = lines02
func_dict['trap01'] = trap01
func_dict['trap02'] = trap02
func_dict['mxn'] = mxn_problem
func_dict['2x4+12'] = p_2x4p12
func_dict['beale'] = beale
func_dict['chung-reynolds'] = chung_reynolds
func_dict['dixon-price'] = dixon_price
func_dict['el-attar-vidyasagar-dutta'] = el_attar_vidyasagar_dutta
func_dict['matyas'] = matyas
func_dict['powell-singular'] = powell_singular
func_dict['powell-singular-2'] = powell_singular_2
func_dict['powell-sum'] = powell_sum
func_dict['rosenbrock'] = rosenbrock
func_dict['rotated-ellipse'] = rotated_ellipse
func_dict['rotated-ellipse-2'] = rotated_ellipse_2
func_dict['rump'] = rump
func_dict['scahffer-1'] = scahffer_1
func_dict['scahffer-2'] = scahffer_2
func_dict['scahffer-3'] = scahffer_3
func_dict['scahffer-4'] = scahffer_4
func_dict['sphere'] = sphere
func_dict['schwefel-1-2'] = schwefel_1_2
func_dict['schwefel-2-21'] = schwefel_2_21
func_dict['schwefel-2-22'] = schwefel_2_22
func_dict['step-1'] = step_1
func_dict['step-2'] = step_2
func_dict['step-3'] = step_3
func_dict['stepint'] = stepint
func_dict['type-1-deceiptive'] = type_1_deceiptive
func_dict['type-2-deceiptive'] = type_2_deceiptive
func_dict['type-3-deceiptive'] = type_3_deceiptive


def get_availabe_funcs():
    my_keys = func_dict.keys()
    print("Available Functions:")
    cont = 1
    for ky in my_keys:
        print(f"Function {cont}: {ky}")
        cont += 1
    return my_keys


def main(argv):
    print(argv)
    print(f"Hello, I currently have {len(func_dict.keys())} functions")
    x = []
    my_func = None
    my_func_name = ''
    m = 0
    for y in argv[0]:
        x.append(int(y))
    xi = np.array(x)
    if len(argv) > 1:
        m = int(argv[1])
    if len(argv) > 2:
        my_func_name = argv[2]
    else:
        print('Please specify a function')
        return
    get_availabe_funcs()
    my_func = func_dict[my_func_name]
    print('function is ', my_func)
    print('xi = ', xi)
    q = int(len(x)/m)
    alphas = [random.random() for i in range(q)]
    print(f"alphas ={alphas}")
    xi = [3, 0.5]
    alphas = [0, 0, 0, 0]
    print(f"xi ={xi}")
    y = my_func(xi, beta=1, my_m=m, alphas=alphas, modo='real')
    print(y)


if __name__ == '__main__':
    main(sys.argv[1:])

#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
if __name__ == '__main__':
    main(sys.argv[1:])

#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
#python3 test_functions.py 1111000011110000 4 type-3-deceiptive
