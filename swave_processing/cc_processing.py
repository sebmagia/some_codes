import numpy as np
import os
from scipy import interpolate
from scipy.special import jv,jn_zeros,j0
from scipy.optimize import curve_fit
import scipy.sparse as scsp
from scipy.signal import welch, csd, tukey
from scipy.linalg import block_diag
from scipy.sparse.linalg import bicg,cg,LinearOperator,aslinearoperator
from functools import partial
from itertools import permutations,combinations,product
import obspy


"""
Collection of functions for obtaining a dispersion curve from cross-coherence data. Based on Menke and Jin (2015)
and Olivar-Castaño et al., (2020). 
"""

def cross_coherence(st,fs,s,stations,norm=None,onesided=True):
    """
    This function takes two Stream objets from obspy and performs the cross-coherence with the Welch Method
    """
    sig1=st.select(station=stations[0])[0]
    sig1.filter('highpass',freq=0.9,corners=2)
    sig2=st.select(station=stations[1])[0]
    sig2.filter('highpass',freq=0.9,corners=2)
    
    size=int(fs*s)
    step=int(size/2)
    ## alpha=0.1 yields 5% taper at each end
    W=tukey(size,alpha=0.1)

    if norm == '1bit':
        ## apply one-bit normalization
        sig1=np.sign(sig1)
        sig2=np.sign(sig2)

    

    Pxy = csd(sig1, sig2, fs=fs, window=W, noverlap=step, detrend='linear', return_onesided=onesided)
    Pxx = welch(sig1, fs=fs, window=W, noverlap=step, detrend='linear', return_onesided=onesided)
    Pyy = welch(sig2, fs=fs, window=W, noverlap=step, detrend='linear', return_onesided=onesided)

   
    Cxyp = ((1 / len(Pxy[0])) * np.real(Pxy[1]) / np.sqrt(((1 / len(Pxy[0])) ** 2) * np.real(Pxx[1]) * np.real(Pyy[1])))
    print(Cxyp.shape)
    f=Pxy[0]
    if onesided:
        Cxy=Cxyp
    else:
        f=f[:len(f)//2]
        data_sp = np.split(Cxyp, 2)
        Cxy = (data_sp[0] + data_sp[1][::-1]) / 2
    return f,Cxy,Cxyp


def get_dispersion_curve2_gs(f,CC,r,smooth_factor,f1,f2,chigh,clow,ii,coord_uni,p12,create_figures=False):
     """

    this function performs a grid search algorithm to retrieve a dispersion curve from CC data.
    it then fits a custom function to the grid search estimate that more accurately describes the expected
    shape of dispersion curves.

    w: angular frequency vector
    CC: observed CC waveform (not smoothed)
    r: interstation distance
    f1,f2: lower and upper freq. limits where to perform the grid search
    chigh,clow: upper and lower velocity limits where to perform the grid search
    """

    w=2*np.pi*f
    wmin=w[0]
    wmax=w[-1]
    ## smooth the CC waveform with a moving-average filter
    CC_smooth=smooth(CC,smooth_factor)
    ## Define window where calculate zero_crossings
    idx=np.where((f >=f1) & (f <=f2))[0]
    fx=f[idx]
    wx=fx*2*np.pi
    CCX_smooth=CC_smooth[idx]
    CCX=CC[idx]

    wmin=wx[0]
    wmax=wx[-1]
    NN=4
    ## generate a log-spaced vector of NN frequency points where the grid search is performed 
    ## using NN larger than 5 will unnecesarrily increase the grid search execution
    wtrial=np.logspace(np.log10(wmin),np.log10(wmax),NN)
    ftrial=wtrial/(2*np.pi)
    ## define the grid of phase velocities. NC larger than 40 will unnecessarily increase the 
    ## grid search execution
    
    cgrid=np.linspace(chigh,clow,NC)
    ## generate all monotonic decreasing possible combinations without repetition
    c_combs=np.array(list(combinations(cgrid,NN)))
    ## split ccombs to improve performance. Number of possible split depends on NC and NN!
    ccombs2=np.split(c_combs,10,axis=0)
    best_sol=1000000

   
    
    for x in ccombs2:
            
            ip=interpolate.interp1d(wtrial,x,'linear',fill_value='extrapolate')
            ipw=ip(wx).astype(np.float32)
            rho_pre=j0(wx*r/ipw)
            A_est = (rho_pre @ CCX_smooth)/(np.linalg.norm(rho_pre,axis=1))**2
            A_est_t=A_est.reshape(len(A_est),1)
            rho_pre=np.multiply(A_est_t, rho_pre)
            Drho=CCX_smooth-rho_pre
            err=np.linalg.norm(Drho,axis=1)**2
            agmin=np.argmin(err)
            Asol=A_est[agmin]
            rhosol=rho_pre[agmin]
            errsol=err[agmin]
            csol=ipw[agmin]
            ccombsol=x[agmin]
            if err[agmin]<best_sol:
                best_sol=err[agmin]
                agmin=np.argmin(err)
                A_best=Asol
                rho_best=rhosol
                err_best=errsol
                c_best=csol
                ccombs_best=ccombsol
    ## we discard data points where condition lambda/r >= 0.45 does not fulfill
    lambdar=0.45
    cond=np.where( ccombs_best/ftrial/r >= lambdar)
  
    ## update initial estimate with the custom function with a nonlinear least squares algorithm
    popt,pcov=curve_fit(tanhfit,wtrial,ccombs_best)
    cbest_int=tanhfit(wx,*popt)
    
    rho_best=jv(0, wx*r/(cbest_int))
    A_best = (np.sum(rho_best *CCX_smooth))/(np.linalg.norm(rho_best))**2
    rho_best=A_best*rho_best
   
   
    return fx,CCX,CCX_smooth,c_combs,ftrial,wtrial,A_best,rho_best,err_best,ccombs_best,cbest_int
    
def get_dispersion_curve_newton(w,r,rho_0,rho_obs,f_full,c0,A0,sigma_D,sigma_A,alpha,
                                ii,nfold,coord_uni,p12,wzero,zcs,CDINV,folder,med,save=False):

    """
    this function takes the initial estimate from the grid search and improves it with a newton method. 
    it also computes a measure of the inversion uncertainty by means of the posterior covariance matrix.

    w: angular frequency vector
    r: interstation distance
    rho_0: computed CC function from the initial estimate phase velocity dispersion curve c0
    rho_obs: (smoothed) measured CC (i.e., the target function)
    c_0: initial estimate phase velocity dispersion curve
    A_0: initial estimate of the attenuation parameter
    sigma_D: variance assigned to the data
    sigma_A: variance assigned to the initial estimate
    alpha: parameter that weighs the degree of smoothing of the dispersion curve
    """
    wmin=w[0]
    wmax=w[-1]
    N=len(w)
    Drho=rho_obs-rho_0
    E=Drho@Drho
    cp=c0
    Ap=A0
    rhop=rho_0
    Ntop = N + 1
    Nbot = N - 2
    
    ## H Matrix with smallness and smoothness constraints
    H = np.zeros((Ntop + Nbot, N + 1))
    #smallness
    H1 = sigma_A * np.eye(Ntop)
    # smoothness
    dw=w[1]-w[0]
    H2 = np.zeros((N, N))
    for i in range(1, len(H2) - 1):
        H2[i, i - 1] = 1
        H2[i, i] = -2 
        H2[i, i + 1] = 1
    H2=alpha*H2
    H[:N + 1, :N + 1] = H1
    H3 = H2[1:-1]
    H[N + 1:, :-1] = H3
    h = np.zeros((Ntop + Nbot))
    H = scsp.csr_matrix(H)
    HH = H.T @ h

    # auxiliar function for the newton method
    f = lambda x: wlstsq(x, G, H)
    ## transform the data covariance matrix to sparse matrix so it uses less space
    CDINV=scsp.csr_matrix(CDINV)

    ## number of interations until newton method stops. 25 should be fine
    niter = 25

    for iter in range(niter):
        Drho0=Drho

        ## G1 and G2 are the Data kernel composed by the parameter derivatives (Menke and Jin 2015). 
        G1 = scsp.lil_matrix(np.diag(Ap * jv(1, w * r / cp) * w * r * cp ** (-2)))
        G2 = scsp.lil_matrix(jv(0, w * r / c0)).transpose()
       
        G = CDINV@scsp.hstack((G1, G2)).tocsr()
      
        F=G.T@Drho+HH

       
        L = LinearOperator((G.shape[1], G.shape[1]), matvec=f, rmatvec=f)
        Dm = cg(L, G.T @ Drho + HH, tol=1e-08, maxiter=4 * N)

        cp = cp + Dm[0][0:N]
        Ap = Ap + Dm[0][N]
        rhop = Ap * jv(0, w * r / cp)
        Drho = rho_obs - rhop
        
    ## update G to generate posterior covariance matrix
    G1 = scsp.lil_matrix(np.diag(Ap * jv(1, w * r / cp) * w * r * cp ** (-2)))
    G2 = scsp.lil_matrix(jv(0, w * r / c0)).transpose()
    G= CDINV * scsp.hstack((G1, G2)).tocsr()
    # compute posterior covariance matrix
    CM=scsp.linalg.inv(scsp.vstack((G,H)).T@scsp.vstack((G,H))).todense()
    
    err=np.abs(rho_obs-rhop)
    
    return rhop,cp,Ap,CM,err,G,G1,G2

          

def wlstsq(m,G,H):
    return G.T @ G @ m + H.T @ H @ m


def tanhfit(omega,d,e,f):
    c=d/np.tanh(e*omega)+f/np.sqrt(omega)
    return c

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth