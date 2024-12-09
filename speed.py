# import cvxpy as cp

import numpy as np
import random
import math
import copy
import time

def func_fn(x,index):
    return pow(x,3) + (T*H[index]/(s*G[index])-E[index]/G[index])*x - F[index]*H[index]/(s*G[index])
def func_fn_ex(x,index):
    return pow(x,3) + (T*p[index]/(G[index])-E[index]/G[index])*x - F[index]*H[index]/(s*G[index])
def func_bn(x,index):
    return s/(x*np.log2(1+A[index]/x))+F[index]/f[index] - T#x*np.log2(1+A[index]/x) - s/(T-F[index]/f[index])
def func_bn_E(x,index):
    return G[index] * f[index] * f[index] + H[index]/(x*np.log2(1+A[index]/x)) - E#x*np.log2(1+A[index]/x) - s/(T-F[index]/f[index])
    
def generate_shadow_fading(mean,sigma,size):
    sigma=pow(10,sigma/10)
    mean=pow(10,mean/10)
    m = np.log(pow(mean,2)/np.sqrt(pow(sigma,2)+pow(mean,2)))
    sigma= np.sqrt(np.log(pow(sigma,2)/pow(mean,2)+1))
    Lognormal_fade=np.random.lognormal(m,sigma,size)
    return Lognormal_fade
def binary(func,convergence, left, right,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  
    cur_root = left
    count = 1
    while error > convergence:
        if abs(func(left,index)) < convergence:
            return left
        elif abs(func(right,index)) < convergence:
            return right
        else:
#             print(str(count) + ' root = ' +str(cur_root))
            middle = (left + right) / 2
            if (func(left,index) * func(middle,index)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(func(cur_root,index))
        count += 1
        if count > 1000:
            #print('There is no root!')
            return cur_root
    return cur_root
def find_T():
    T_min = 0.0001#max(F/f_max)
    T_max = 5
    T_max_fixed = 20
    T = 3
    while True:
        f_list = []
        b_list = []
        tola = 0.00001
        for i in range(n):
            f_temp = binary(func_fn,1e-3, f_min, f_max,index = i)
            f_list.append(f_temp)
        f = np.array(f_list)
        for i in range(n):
            b_temp = binary(func_bn,1e-2, 0.0001, coef * B ,index = i)
            b_list.append(b_temp)
        b = np.array(b_list)#A/(T-F/f)
        b_sum = sum(b)
        ratio_b = b_sum/B
#         print(ratio_b)
#         print(T)
        if min(b) < 0:
            T_min = T
            T = (T+T_max)/2
            continue
        if (1-tola) <= ratio_b <= 1+tola:
            T_out = T
            return 
        elif ratio_b < 1 - tola:
            T_max = T#(T + T_min)/2
            #print(2)
            T = (T + T_min)/2
        elif ratio_b > 1:
            T_min = T
            #print(1)
            T = (T_max + T)/2
        
def check_p_bound():
    p_list = []

    for i in range(num_clients):
    #     print(i)
        p_temp = binary(func1, convergence = 1e-3, left = p_min, right = p_max, index = i)
        p_list.append(p_temp)
    #    print(p_temp)
    return np.array(p_list)
def func1(x,index):
    return s*x/(np.log2(1+g[index]*x/(N0 * b[index]))*b[index])-E[index] + G[index] * f[index] * f[index]
num_clients = 10;
n = num_clients
dist = (10+250)*np.random.random(num_clients)*0.001 #250*1.414*np.random.random(50);# 
dist1 = 50*0.001
dist0 = 10*0.001

# dist = np.loadtxt('distance.txt')


plm = 128.1 + 37.6*np.log10(dist);

rd_number = generate_shadow_fading(0,8,n)

PathLoss_User_BS=plm+rd_number

g= pow(10,(-PathLoss_User_BS/10))

N0=-174#;%dBm/Hz
N0=pow(10,(N0/10))/1e3;





i = np.array([i for i in range(0,num_clients)])
C = (2e4/49)*i + 1e4
random.shuffle(C)


# C = np.loadtxt('C')
dist = np.loadtxt('./dist_T_10_big.txt')
C = np.loadtxt('./D_n_T_10_big.txt')
g = np.loadtxt('./G_T_10_big.txt')

# Problem data.
B = 20*1e6;#B
f_max = 2*1e9;# computation capacity/CPU frequency
f = np.ones(num_clients) * f_max
f_min = 0.2*1e9;
p_max = 0.2
p_min = 0.01
p = np.ones(num_clients) *p_max;# trans2mission/transmit power

delta = 0.1;#
xi = 0.1;#
epsilon = 0.001;# 
s = 4894444#28800;#UE update size
alpha = 2e-28;#
#N0 = 1e-8;# -174dBm/Hz
D = 500;#data samples
k = 1e-28;

E = 0.4*np.ones(num_clients)#(0.001+0.002 * np.random.random_sample([10])).tolist()#0.002#
# E_list = 0.5#np.ones(num_clients) * E
T0 = 100;#
b_init= np.ones(num_clients) * (B/num_clients)
b = b_init

eta = 0.5;
gamma = 1;#
l = 1;#
a = (np.log2(1/epsilon)*2*pow(l,2))/(gamma*gamma*xi)
v = 2/((2-l*delta)*delta*gamma)

Ik = v*np.log2(1/eta)
I0 = a/(1-eta)

A = g * p / N0#s / np.log2(1+g*p/(N0*b)) 
F = v * C * D *np.log2(1/eta) 
G = v  * k * C * D * np.log2(1/eta)
H = (s * p)#/np.log2(1+g*p/(N0*b)) 

T_min = 0.0001#max(F/f_max)
T_max = 1
T_max_fixed = 1
T = 1


count = 0 
ratio_b  = 0
tola = 0.01
start = time.time()
while not (1-tola <= ratio_b and ratio_b <=1):
    count += 1
    f_list = []
    b_list = []

    for i in range(n):
        f_temp = binary(func_fn,1e-2, 2e8, 2e9,index = i)
        f_list.append(f_temp)
    f = np.array(f_list)
    for i in range(n):
        b_temp = binary(func_bn,1e-2, 0.01, 3 * B / n ,index = i)
        b_list.append(b_temp)
    b = np.array(b_list)#A/(T-F/f)
    b_sum = sum(b)
    
    ratio_b = b_sum/B

    if ratio_b < 1 - tola:
        T_max = T
        T = (T + T_min)/2
    elif ratio_b > 1:
        T_min = T
        T = (T_max + T)/2
end = time.time()
yongshi = end - start
print(yongshi)