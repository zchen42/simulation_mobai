import scipy
import torch
inf = 1e10+0.0
def LP(U, omega): #U.shape = (M,K)
    M,K = U.shape  

    # construct c:
    c =  torch.zeros(K+1)
    c[K]=1.0

    # construct Aeq and Beq:
    Aeq = torch.ones(K+1) 
    Aeq[K]=0.0
    Aeq = Aeq.view(1,-1)
    Beq = torch.tensor(1)
    Beq = Beq.view(1,1)

    # construct LB and UB:
    LB = torch.zeros(K+1) 
    LB[K]=-inf
    UB = torch.ones(K+1) 
    UB[K]=inf

    # construct Aub and Bub:
    Aub = []
    Bub = []
    for m in range(M):
        u = U[m]
        i_star = u.argmax()
        for i in range(K):
            if i == i_star:
                continue
            delta = u[i_star] - u[i]
            ngla = torch.zeros(K+1)
            ngla[i_star] = torch.square(delta)/2 * torch.square(omega[i])/torch.square(omega[i]+omega[i_star])
            ngla[i] = torch.square(delta)/2 * torch.square(omega[i_star])/torch.square(omega[i]+omega[i_star])

            bub = -torch.square(delta)/2*(omega[i]*omega[i_star])/(omega[i]+omega[i_star]) + ngla[i_star] * omega[i_star] + ngla[i]*omega[i] # -g + <ngla,omega>
            ngla[K] = 1
            Aub.append(-ngla)
            Bub.append(-bub.view(1))
    Aub = torch.stack(Aub)
    Bub = torch.cat(Bub)
    bounds = torch.cat([LB.view(-1,1), UB.view(-1,1)], dim=1)
#print(bounds)
    ret =scipy.optimize.linprog(c=c, A_ub=Aub, b_ub=Bub, A_eq=Aeq, b_eq=Beq, bounds=bounds)
    return ret['x'][:K]
