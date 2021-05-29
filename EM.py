def KF(x0, P0, F, Q, H, R, zk):
    x_pred = F*x0            #state prediction
    P_pred = F*P0*F + Q      #state prediction covariance
    z_hat  = H*x_pred        #measurement prediction
    inov   = zk - z_hat      #inovation
    S      = R + H*P_pred*H  #innovation covariance
    G      = P_pred*H/S      #kalman gain
    xkk    = x_pred + G*inov #state update
    Pkk    = P_pred - G*S*G  #covariance update
    return [xkk, Pkk, x_pred, P_pred, G]
def KS(x0n, P0n, xkk, Pkk, xk_pred, PK_pred, F, H, Kgain):
    i = len(xkk)-2; C = []
    Xkn = xkk.copy(); Pkn = Pkk.copy()
    while i>=0:
        temp = Pkk[i]*F/PK_pred[i]
        C.insert(0,temp)
        Xkn[i] = xkk[i]+temp*(Xkn[i+1]-xk_pred[i+1])
        Pkn[i] = Pkk[i]+temp*temp*(Pkk[i+1]-PK_pred[i+1])
        i -= 1
    C0  = P0n*F/PK_pred[0]
    x0n = x0n +C0*(Xkn[0]-xk_pred[0])
    P0n = P0n +C0*C0*(Pkn[0]-PK_pred[0])
    i = len(xkk)-2
    PL1 = []
    temp = (1-Kgain*H)*F*Pkk[-2]
    PL1.insert(0,temp)
    i -=1 
    while i>=0:
        temp=Pkk[i-1]*C[i-1] + C[i]*(temp - F*Pkk[i-1])*C[i-1]
        PL1.insert(0,temp)
        i -= 1
    PL0 = P0n*C0+C[0]*(PL1[0]-F*P0n)*C0
    return [Xkn,Pkn,x0n,P0n,PL1,PL0]
def EM(x0n,P0n,xkn,Pkn,PL1,PL0,z):
    S10 = xkn[0]*x0n + PL0
    S00 = x0n*x0n+P0n
    zkx = z[0]*xkn[0]
    xkp = xkn[0]*xkn[0] + Pkn[0]
    for i in range(1,len(xkn)-1):
        S10 = S10 + xkn[i]*xkn[i-1] + PL1[i-1]
        S00 = S00 + xkn[i-1]*xkn[i-1] + Pkn[i-1]
        zkx = zkx + z[i]*xkn[i]
        xkp = xkp + xkn[i]*xkn[i] + Pkn[i]
    F = S10/S00
    H = zkx/xkp
    return(F,H)
x0 = 2; P0 = 1; F  = 0.5; Q  = 0.25; H  = 1; R  = 0.25; flag = True
zk = [1.1,1.5,1.2,1,0.8,1.5,1.9,1.3,1.2,0.7,0.8,1.2,1.5]
for count in range(1,5):
    xkk = []; Pkk = []; x_pred = []; P_pred = []
    flag = True
    for i in zk:
        if flag == True:
            [xk,Pk,x_pre,P_pre,G] = KF(x0,P0,F,Q,H,R,i)
            flag = False
        else:
            [xk,Pk,x_pre,P_pre,G] = KF(xk,Pk,F,Q,H,R,i)
        xkk.append(xk)
        Pkk.append(Pk)
        x_pred.append(x_pre)
        P_pred.append(P_pre)
    [xkn,Pkn,x0n,P0n,PL1,PL0] = KS(x0,P0,xkk,Pkk,x_pred,P_pred,F,H,G)
    [F,H] = EM(x0n, P0n, xkn, Pkn, PL1, PL0, zk)
    x0 = x0n; P0 = P0n
    print(F,H)
