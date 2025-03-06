import torch
import doLP
import math
import decimal

inf = 1e10+0.0
def pull_arm(i):
    reward = torch.normal(U[i], std[i])
    N[i] = N[i] + 1
    emprical_mean[i] = emprical_mean[i]*(N[i]-1)/N[i] + reward/N[i]
    Buffer[i] = Buffer[i] - 1
    return 0
def main(confidence_level):
    global U, Buffer,std, emprical_mean,N


    K,M = U.shape  
    std = torch.ones(K,M)

    Buffer = torch.zeros(K,dtype=torch.float64)
    omega_hat = [1/K for i in range(K)]
    omega_hat = torch.tensor(omega_hat)
    Nt = K
    N = torch.ones(K)
    emprical_mean = torch.normal(U,std)
    last_mean = emprical_mean.clone()
    t = K+1
    eta= 0.1
    while True: 
        if (N.min() -1 <=math.sqrt(t/K)) or (N.min() -1 < t/K *eta): 
            At = N.argmin()
            pull_arm(At)
        else:
            if t >= 0:
                last_mean = emprical_mean.clone()
            s = doLP.LP(last_mean.T, omega_hat)
            Nt = Nt+1
            Buffer = Buffer + s
            omega_hat = (Nt-1)/Nt*omega_hat + s/Nt
            At = Buffer.argmax()
            pull_arm(At)
        Z_t = torch.tensor(inf)
        for m in range(M):
            u = emprical_mean.T[m]
            i_star = u.argmax()
            for i in range(K):
                if i == i_star: 
                    continue
                delta = u[i_star] - u[i]
                Z_t = torch.min(Z_t, torch.square(delta)/2/(N[i]+N[i_star])*N[i]*N[i_star])

        if  Z_t-math.log(math.log(t)+1)> -math.log(confidence_level):
            break
        t =t +1
    return t,emprical_mean

confidence_level = float(input('input confidence_level: '))

K = int(input('input K='))
M = int(input('input M='))
U = torch.randn(K,M)
for k in range(K):
    for m in range(M):
        U[k][m] = float(input("input the mean of arm %d in objetive %d: "%(k+1,m+1)))

print('running...')

t,emprical_mean = main(confidence_level)
print("t=",t)
for m in range(M):
    u = emprical_mean.T[m]
    i_star = u.argmax()
    print('The best arm of objective %d is %d'%(m+1,i_star+1))

