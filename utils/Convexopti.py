import cvxpy as cp
import numpy as np
import random
import math
from scipy.optimize import rosen, rosen_der
from ipopt import minimize_ipopt
from math import log,ceil

def binary(func,convergence, left, right,index = None):
#     print('current acceptable error: ' + str(convergence) + '\n')
    error = convergence + 1  # 循环开始条件
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
    return cur_root

def maxs(a, b_array):
    length = len(b_array)
    alist = []
    for i in range(length):
        b = b_array[i]
        alist.append(((a+b) + abs(a-b))/2)
    return np.array(alist)

# func1 用于求eta*
def func1(x,index):
      return sum(X1*(1-1/x)/np.log(2)-X1*np.log2(x)+X2)
# func2 用于求eta_min与eta_max
def func2(x,index):
      return X4[index]*(1-1/x)/np.log(2)-X4[index]*np.log2(x)+X3[index]
    
def func3(x,index):
      return (X3[index]+X4[index]*np.log2(1/x))/(1-x)-T0


class convex_opti(object):
    def __init__(self, num_clients,B,f_max,f_min,p,T0):        
        self.B = B
        self.f_max = f_nax
        self.f_min = f_min
        self.p = p
        self.num_clients = num_clients
        self.T0 = T0
        
        self.L0 = 140.7
        self.dist = (250*1.414)*np.random.random(50)*0.001 #250*1.414*np.random.random(50);# 在500*500正方形区域内
        self.dist1 = 50*0.001
        self.dist0 = 10*0.001   
        
        PL = []
        for i in range(num_clients):
            dist_temp = dist[i]
            if dist_temp > dist1:
                PL.append(-L0 - 35*np.log10(dist_temp))
            elif dist1 >= dist_temp > dist0:
                PL.append(-L0 - 15*np.log10(dist1)- 20*np.log10(dist_temp))
            else:
                PL.append(-L0 - 15*np.log10(dist1)- 20*np.log10(dist0))

        z = np.random.random(num_clients)*8
        PL = np.array(PL)
        beta = pow(10,self.PL/10)*pow(10,z/10)
        self.g = np.sqrt(beta)*z
        i = np.array([i for i in range(0,num_clients)])
        self.C = (2e4/49)*i + 1e4#C:1~3℅10^4               
        random.shuffle(self.C)
               
        self.delta = 0.1;#δ
        self.xi = 0.1;#ξksi
        self.epsilon = 0.001;# ε
        self.s = 20000;#UE update size
        self.alpha = 2e-28;#α
        self.N0 = 1e-8;# -174dBm/Hz
        self.D = 500;#data samples
        self.k = 1e-28;
        self.b_init= self.B/self.num_clients;
        self.b= self.b_init * 1e6
        self.f= ((f_max+f_min)/2)*1e9
        gamma = 2;#γ-strongly
        l = 2;#l-Lipschitz
        self.a = (np.log2(1/epsilon)*2*pow(l,2))/(gamma*gamma*xi)
        self.v = 2/((2-l*delta)*delta*gamma)
        
    def optimizing():
        count_time = 0
        while count_time < 10:
            X1 = self.a*self.v*self.k*self.C*self.D*pow(self.f,2)
            X2 = (self.a*self.p*self.s)/(np.log2(1+(self.g*self.p)/self.N0)*self.b)
            X3 = (self.a*self.s)/(np.log2(1+(self.g*self.p)/self.N0)*self.b)
            X4 = (self.a*self.v*self.C*self.D)/self.f

            eta0_list = []
            for i in range(num_clients):
            #     print(i)
                eta0_temp = binary(func2, convergence = 1e-6, left = 0.00001, right = 1, index = i)
                eta0_list.append(eta0_temp)

            min_list = []
            max_list = []
            for i in range(num_clients):
            #     print(i)
                eta_temp_min = binary(func3, convergence = 1e-6, left = 0.00001, right = eta0_list[i], index = i)
                eta_temp_max = binary(func3, convergence = 1e-6, left = eta0_list[i], right = 1,index = i)
                min_list.append(eta_temp_min)
                max_list.append(eta_temp_max)

            #删除列表中的None值
            max_list = dele_none(max_list)
            min_list = dele_none(min_list)

            eta_max = min(max_list)
            eta_min = max(min_list)

            # 求最佳eta
            eta = binary(func1, convergence = 1e-6, left = 0.00001, right = 1)
            if (eta_min >= eta):
                eta = eta_min
            elif eta_max <= eta :
                eta = eta_max
            if eta_max <= eta_min:
                eta = (eta_min+eta_max)/2


            A = (self.a*self.v*self.k*self.C*self.D*np.log2(1/eta))/(1-eta)*1e18
            F = (self.s*self.a*self.p)/((np.log2(1+(self.g*self.p)/N0))*(1-eta))*1e-6
            H = (A/self.k)
            G = (F/self.p)
            J = 10*A/self.T0
            L = F/(self.p*self.T0)
            bn_min = L*(1+J/(self.f_max-J))
            bn_max = L*(1+J/(self.f_min-J))
            bn = cp.Variable(self.num_clients)
            
            ob_func = cp.sum(A @ cp.power(J*(1+L@cp.inv_pos(bn-L)),2)*cp.power(J,2)+F@cp.inv_pos(bn))
            objective = cp.Minimize(ob_func)
            constraints = [cp.sum(bn) <= self.B,
                          bn >= bn_min,
                          bn <= bn_max]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            if bn.value.any == None:
                print('Stop the iteration')
                break
                
            self.b  = (bn.value)*1e6
            self.f = J*(1+L/(bn.value-L))* 1e9
            count_time += 1
        return self.b, self.f, eta