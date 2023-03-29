import numpy as np
from scipy.optimize import nnls,minimize,minimize_scalar,fmin,basinhopping,Bounds
import timeit
from scipy.sparse import block_diag

 """
I extracted this functions from Seismic_Cycle class, so they need changes to work on their own.

Functions for performing a GNSS+InSAR bayesian inversion of the coseismic slip distribution
that maximizes the bayesian evidence. Solutions are semianalytical and fast to compute.

Based on Benavente et al., 2019: Efficient Bayesian uncertainty estimation in linear finite fault inversion with positivity
constraints by employing a log-normal prior 
for  Sebastián Núñez-Jara Bachelor thesis (2021): UN ENFOQUE BAYESIANO PARA LA INVERSIÓN DE DATOS GEODÉSICOS Y SELECCIÓN DE MODELOS
APLICADO AL DESLIZAMIENTO COSÍSMICO ASOCIADO AL TERREMOTO DEL MAULE M W 8,8 DE 2010.
 """
 def bayesian_inversion_new(self,**kwargs):
        """ we invert the coseismic data  and obtain the slips in every subfault
        , with these slips, we generate synthethic displacements in surface and compute
        the mean cuadratic error"""

        self.paper=kwargs.get('paper','yabuki')
        lon = [elem.get('Longitud') for elem in self.data_estaciones]
        lat = [elem.get('Latitud') for elem in self.data_estaciones]
        sta = [elem.get('Station') for elem in self.data_estaciones]
        nx=self.grid_size[0]
        ny=self.grid_size[1]
        ## we generate always the same numbers!
        seed=550000
        np.random.seed(seed)
        ## choose between maxevi, fixhyper and linear inversions. maxevi is for the thesis.
        ## it defines another parameter weighting insar and gnss data (see agata paper)
        formulation=kwargs.get('formulation','maxevi')
        ## choose between GNSS and INSAR or joint datasets 
        data_type=kwargs.get('data_type','GNSS')
        ## choose covariance matrix type:
        ## options for GNSS: 'data': load data uncertainties and create covariance matrix from these (useful 4 inversion of real data)
        ##                   'noise': add noise to data and use that 
        ##                    noise as covar matrix (useful 4 synthetic test)

        cd_gnss_type=kwargs.get('cd_gnss','data')
        cd_insar_type=kwargs.get('cd_insar','corr')
        ## JOINT INSAR + GNSS INVERSION
        if data_type=='JOINT':
            self.data_type='JOINT'
            ## cd insar without std
            CD_insar=kwargs.get('CD_insar',None)
            gama=kwargs.get('gama',1.0)
            ## CD insar with std and without gamma
            data,CD,CD_GNSS,CD_INSAR=self.data_inversion(data='JOINT',
                                                        cdi='corr',
                                                        CD_insar=CD_insar,
                                                        gama=gama)
            ## loading GNSS and InSAR green functions for inversion
            Ggps=kwargs.get('G',None)
            Glos=kwargs.get('Glos',None)
            G=np.vstack((Ggps,Glos))
       
         ## load cholesky descomposition of prior covariance matrix
        L=kwargs.get('L',None) 

       
        print('minimizing the objective function')
       
        if self.interface=='single':
                    
                ## obs: G.T@d es un buen guess para m
                map_guess=G.T@data

        if self.interface=='double':
                map_guess=G.T@data

        if formulation == 'maxevi':

            if self.paper=='yabuki':
                method='Bounded'
                bounds=(10**(-2),10**6)
                bounds=(0.3,0.4)
                
            self.soldict={}
            self.ABICSOL=[]
            self.initial_guess=map_guess
            start=timeit.default_timer()
           
            args=(data,G,np.linalg.inv(CD),L)

          
            alpha_sol=minimize_scalar(self.abic_alpha_nonlinear,
                                      args=args,
                                      method=method,
                                      bounds=bounds)
            self.alpha_sol=alpha_sol
                                      #options={'maxiter':1000,'xatol':1e-05})

           
            print(alpha_sol)
            sol_raw=self.soldict.get('sol')
            sol=sol_raw.x
            alpha_sq=alpha_sol.x
            stop = timeit.default_timer()
            print('TIMER')
            print('Time: ', stop - start)

            print('RESULTADOS')
            print(alpha_sol)
            print('map_result',map_guess)
            print('exp_map',np.exp(map_guess))
            print('alpha_sol',alpha_sol)
            print('L',L)
            print('sol_raw',sol_raw)
        
        ## calculamos evidencia, covarianza posterior y todo
        ## el postprocesamiento con la hessiana
       
           
        hess=self.hessian(sol_raw.x,G,data,L,np.linalg.inv(CD),alpha_sq)

        ## compute the logarithm of the evidence and hyperparameter sigma
        logevi,sigma_sq=self.evidence_alpha_sigma_nonlinear(sol_raw.x,
                                                            sol_raw.fun,
                                                            alpha_sq,
                                                            data,
                                                            G,
                                                            CD,
                                                            L,
                                                            hess)
            
            
        rho_sq=sigma_sq/alpha_sq
        print('calculating posterior covariance metrix')

        postco=2*sigma_sq*np.linalg.pinv(hess)

     

                    
        
            ## check whether hessian matrix is positive definite
            ## a covariance matrix MUST be positive definite 
        try:
            b=np.linalg.cholesky(hess)
            print('hessian matrix is positive definite')
        except np.linalg.LinAlgError:
            print('WARNING: HESSIAN MATRIX IS NOT POSITIVE DEFINITE')

            
            
        ## calculate marginal pdf for each subfault
        marg=self.get_marginal(sol_raw.x,postco,hist=False)

        print('generating synt. data from solution')
        if data_type=='GNSS':
            dsint=G@np.exp(sol_raw.x)
            uenz_sint=np.split(dsint,3)
            Ue_sint=1000*uenz_sint[0]
            Un_sint=1000*uenz_sint[1]
            Uz_sint=1000*uenz_sint[2]
            self.inverted_data.update({'UE':Ue_sint,
                                        'UN':Un_sint,
                                        'UZ':Uz_sint})
        if data_type=='JOINT':
            dsint_gps=Ggps@np.exp(sol_raw.x)
            dsint_insar=Glos@np.exp(sol_raw.x)
            dsint=G@np.exp(sol_raw.x)
            uenz_sint=np.split(dsint_gps,3)

            Ue_sint=1000*uenz_sint[0]
            Un_sint=1000*uenz_sint[1]
            Uz_sint=1000*uenz_sint[2]
            self.inverted_data.update({'UE':Ue_sint,
                                       'UN':Un_sint,
                                       'UZ':Uz_sint,
                                       'Ulos':dsint_insar})

        if data_type=='INSAR' or data_type=='INSAR_JOINT':
            dsint=G@np.exp(sol_raw.x)
            self.inverted_data.update({'Ulos':dsint})


        return alpha_sq,sol,logevi,sigma_sq,CD,L,hess,marg,dsint,data 


def abic_alpha_nonlinear(self,alpha_sq,d,G,Cd_inv,L):
        """
        bayesian inverse problem by maximizing the evidence
        """
        M=np.shape(L)[0]
        P=np.linalg.matrix_rank(L)
        N=len(d)
        funcparams=(G,d,L,Cd_inv,alpha_sq)
        
        sol=minimize(self.obj_nonlinear,
                     self.initial_guess,
                     args=funcparams,
                     method='trust-ncg',
                     jac=self.jacobian2,
                     hess=self.hessian2,
                     options={ 'gtol': 1e-4})
                              
        hess=sol.hess
        if len(sol.x.shape)==1:
            map_g=sol.x.reshape((len(sol.x)),1)
            self.initial_guess=sol.x.reshape((len(sol.x)),1)
       
            
        e=np.exp(map_g)
        if len(d.shape)==1:
            d=d.reshape((len(d)),1)
      
        psi_map=sol.fun
        sigma_sq=psi_map/N
       
        res=G@np.exp(sol.x)
        
        sign,det=np.linalg.slogdet(0.5*hess)
       


        if sign <0:
            print('warning, el signo no debe ser menor que 0')
      
        C=300
        if self.paper=='yabuki':
            ABIC_alpha=(N+P-M)*np.log(psi_map)-P*np.log(alpha_sq)+sign*det+C

        if self.paper=='fuku':
            ABIC_alpha=(N+P-M)*np.log(psi_map)-P/2*(np.log(alpha_sq)+np.log(beta_sq))+sign*det+C

        if self.paper=='benavente':
            ABIC_alpha=psi_map+sign*det+P*np.log(alpha_sq)
       
        self.soldict.update({'sol':sol})
       
        return ABIC_alpha
 def obj_nonlinear(self,s,*args):
        ## objective function to minimize
        ## alpha2 is sigma2/rho2 (different formulation than)
        ## benavente et al., 2019
        ## a guess in alpha and s must be computed in order
        ## to minimize this objective function
        ## this function should be minimum on MAP
        G=args[0]
        d=args[1]
        L=args[2]
        Cd_inv=args[3]
        alpha_sq=args[4]
        
        s=s.reshape(len(L),1)
        d=d.reshape(len(Cd_inv),1)
        try:
            e=np.exp(s)
        except AttributeError:
            print('este error es raro')
        
            
        try:
            ## yabuki
            if self.paper=='yabuki':
               
                psi=(G@e-d).T@Cd_inv@(G@e-d)+ alpha_sq*(L@s).T@(L@s)
                
            ## benavente
            if self.paper=='benavente':
                psi=(G@e-d).T@Cd_inv@(G@e-d)+ alpha_sq**(-1)*(L@s).T@(L@s)
            

           

        except ValueError:
            print('corrigiendo')
            if self.paper=='yabuki':
                psi=(G@e-d).T@Cd_inv@(G@e-d)+ alpha_sq*(L@s).T@(L@s)
            if self.paper=='benavente':
                psi=(G@e-d).T@Cd_inv@(G@e-d)+ alpha_sq**(-1)*(L@s).T@(L@s)




        return psi


    def abic_alpha_nonlinear(self,alpha_sq,d,G,Cd_inv,L):
      
        M=np.shape(L)[0]
        P=np.linalg.matrix_rank(L)
        N=len(d)
        funcparams=(G,d,L,Cd_inv,alpha_sq)
        
        sol=minimize(self.obj_nonlinear,
                     self.initial_guess,
                     args=funcparams,
                     method='trust-ncg',
                     jac=self.jacobian,
                     hess=self.hessian,
                     options={ 'gtol': 1e-4})
                               #'initial_trust_radius':1.0}
        
        hess=sol.hess
        if len(sol.x.shape)==1:
            map_g=sol.x.reshape((len(sol.x)),1)
            self.initial_guess=sol.x.reshape((len(sol.x)),1)
       
            
        e=np.exp(map_g)
        if len(d.shape)==1:
            d=d.reshape((len(d)),1)
        
        sigma_sq=psi_map/N
        
        res=G@np.exp(sol.x)
      
        sign,det=np.linalg.slogdet(0.5*hess)
        

        if sign <0:
            print('warning, el signo no debe ser menor que 0')
       
        C=300
        if self.paper=='yabuki':
            ABIC_alpha=(N+P-M)*np.log(psi_map)-P*np.log(alpha_sq)+sign*det+C

            

        if self.paper=='fuku':
            ABIC_alpha=(N+P-M)*np.log(psi_map)-P/2*(np.log(alpha_sq)+np.log(beta_sq))+sign*det+C

        if self.paper=='benavente':
            ABIC_alpha=psi_map+sign*det+P*np.log(alpha_sq)
      
        self.soldict.update({'sol':sol})
       
        return ABIC_alpha
   
def jacobian(self,s,*args):
        """jacobiano de la funcion objetivo, adapted from benavente et al., 2019 """
        

        G=args[0]
        d=args[1]
        #Cp=args[2]
        L=args[2]
        Cd_inv=args[3]
        alpha_sq=args[4]
        s=s.reshape(len(L),1)
        e=np.exp(s)
        d=d.reshape(len(Cd_inv),1)
        try:
            if self.paper=='benavente':
                jac1=2*np.multiply(e,G.T@Cd_inv@(G@e-d))
                jac2=2*alpha_sq**(-1)*(L.T@L@s)        

            if self.paper=='yabuki':
                jac1=2*np.multiply(e,G.T@Cd_inv@(G@e-d))
                jac2=2*alpha_sq*(L.T@L@s)

                #
        except ValueError:
            print('corrigiendo jacobiano')

            jac1=2*np.multiply(e,G.T@Cd_inv@(G@e-d))
            jac2=2*alpha_sq**(-1)*(L.T@L@s)
          
        jac=jac1+jac2
        if jac.shape[1] ==1:
            #print('reshaping jac')
            jac=jac.reshape(len(jac),)
        return jac
def hessian(self,s,*args,cobain=False):
        """ hessiano de la funcion objetivo benavente et al., 2019  """
        G=args[0]
        d=args[1]
        #Cp=args[2]
        L=args[2]
        Cd_inv=args[3]
        alpha_sq=args[4]
        s=s.reshape((len(L),1))

        e=np.exp(s)
        d=d.reshape(len(Cd_inv),1)
        if self.paper=='benavente':
            hes1=2*np.multiply(e@e.T,G.T@Cd_inv@G)
            hes2=2*np.eye((len(e)))*np.multiply(e,G.T@Cd_inv@(G@e-d))
            hes3=2*alpha_sq**(-1)*L.T@L
        if self.paper=='yabuki':
            hes1=2*np.multiply(e@e.T,G.T@Cd_inv@G)
            hes2=2*np.eye((len(e)))*np.multiply(e,G.T@Cd_inv@(G@e-d))
            hes3=2*alpha_sq*L.T@L
       
        hes = hes1+hes2+hes3
        if not cobain:
            return hes
        if cobain:
            try:
                np.linalg.cholesky(hes1)
            except np.linalg.LinAlgError:
                print('hes1 no es pos. def.')
            try:
                np.linalg.cholesky(hes2)
            except np.linalg.LinAlgError:
                print('hes2 no es pos. def.')
            try:
                np.linalg.cholesky(hes3)
            except np.linalg.LinAlgError:
                print('hes3 no es pos. def.')
            print('H1',hes1)
            print('H2',hes2)
            print('H3',hes3)
            return hes,hes1,hes2,hes3
 def prior_marginal(self,L,alpha_sq):
        mu=0.0
        Cp_inv=L.T@L/alpha_sq
        Cp=np.linalg.inv(Cp_inv)
        diag_std=np.sqrt(np.diag(Cp))
        x=np.linspace(lognorm.ppf(0.02,diag_std[0],scale=1.0),
                                     lognorm.ppf(0.99,diag_std[0],scale=1.0),
                                     1000)
        print(x,diag_std,Cp)
        plt.plot(x,lognorm.pdf(x,diag_std[0],scale=1.0))


   
    
    def evidence_alpha_sigma_nonlinear(self,MAP,obj_map,alpha_sq,d,G,Cd,L,hess,beta_sq=0.0):
        M=np.shape(L)[0]
        P=np.linalg.matrix_rank(L)
        N=len(d)
        Cd_inv=np.linalg.pinv(Cd)
        if self.paper=='yabuki':
            sigma_sq=obj_map/(N+P-M)
            
            ## es P/2 positivo cdo alpha es sigma/rho
            logevi1=0.5*(-N-P+M)*np.log(2*np.pi*sigma_sq)+ (P/2)*np.log(alpha_sq)
            logevi2=np.linalg.slogdet(Cd)
            logevi3=np.linalg.slogdet(L.T@L)
            logevi4=np.linalg.slogdet(0.5*hess)

            if logevi2[0] < 0:
                print('warning, el signo no debe ser menor que 0')
            if logevi3[0] < 0:
                print('warning, el signo no debe ser menor que 0')
            if logevi4[0] < 0:
                print('warning, el signo no debe ser menor que 0')
        
        
            logevi5=-0.5*(sigma_sq)**(-1)*obj_map
           

            ## cuando usamos L.T@L
            logevi=logevi1-0.5*logevi2[0]*logevi2[1]+0.5*logevi3[0]*logevi3[1]+ \
                   -0.5*logevi4[0]*logevi4[1]+logevi5

           
            return logevi,sigma_sq

        if self.paper=='fuku':
            sigma_sq=obj_map/(N+P-M)
            
            logevi1=-0.5*N*np.log(2*np.pi*sigma_sq)+ (M/4)*np.log(alpha_sq)+(M/4)*np.log(beta_sq)
            logevi2=np.linalg.slogdet(Cd)
            L1=np.split(L,2)[0]
            L2=np.split(L,2)[1]
            logevi3=np.linalg.slogdet(L.T@L)
            #
            logevi4=np.linalg.slogdet(0.5*hess)
            if logevi2[0] < 0:
                print('warning, el signo no debe ser menor que 0')
            if logevi3[0] < 0:
                print('warning, el signo no debe ser menor que 0')
            if logevi4[0] < 0:
                print('warning, el signo no debe ser menor que 0')
           
        
            logevi5=-0.5*(sigma_sq)**(-1)*obj_map
          
            ## cuando usamos L.T@L
            logevi=logevi1-0.5*logevi2[0]*logevi2[1]+0.5*logevi3[0]*logevi3[1]+ \
                   -0.5*logevi4[0]*logevi4[1]+logevi5

          
            return logevi,sigma_sq

        if self.paper=='benavente':
            logevi1=0.5*(-N-P+M)*np.log(2*np.pi)-P/2*np.log(alpha_sq)
            logevi2=np.linalg.slogdet(Cd)
            logevi3=np.linalg.slogdet(L.T@L)
            logevi4=np.linalg.slogdet(0.5*hess)
            logevi5=-0.5*obj_map

            ## cuando usamos L.T@L
            logevi=logevi1-0.5*logevi2[0]*logevi2[1]+0.5*logevi3[0]*logevi3[1]+ \
                   -0.5*logevi4[0]*logevi4[1]+logevi5

           
            return logevi

        if self.paper=='agata':
            sigma_sq=obj_map/(N+P-M)
            logevi1=0.5*(-N-P+M)*np.log(2*np.pi*sigma_sq)
            logevi2=np.linalg.slogdet(Cd)
            logevi3=np.linalg.slogdet(beta_sq**(-1)*L.T@L)
            logevi4=np.linalg.slogdet(0.5*hess)
            logevi5=-0.5*(sigma_sq)**(-1)*obj_map
            logevi=logevi1-0.5*logevi2[0]*logevi2[1]+0.5*logevi3[0]*logevi3[1]+ \
                   -0.5*logevi4[0]*logevi4[1]+logevi5
            return logevi,sigma_sq

            
   
    def get_marginal(self,MAP,post_cov,hist=False,prior='log'):
        size=10000
        marginals=np.zeros((len(MAP),size))
        diag_std=np.sqrt(np.diag(post_cov))
        nx=self.grid_size[0]
        ny=self.grid_size[1]
        #print(diag_std[0],MAP[0])
        limx=15
        lin_slip=np.linspace(10**(-3),limx,size)
        log_norms=np.zeros((len(MAP),size))
        log_norms_new=np.zeros((len(MAP),100))
        CIs_new=np.zeros(len(MAP))
        CIs=np.zeros(len(MAP))
        idx1=np.zeros(len(MAP))
        idx2=np.zeros(len(MAP))
        integrals=[]
      
        if prior=='log':
            for i in range(len(MAP)):
                marginals[i,:]=np.random.lognormal(MAP[i],diag_std[i],size)
                lol_slip=np.linspace(lognorm.ppf(0.001,diag_std[i],scale=np.exp(MAP[i])),
                                     lognorm.ppf(0.001,diag_std[i],scale=np.exp(MAP[i])),
                                     100)
                log_norms[i,:]=(lin_slip*diag_std[i]*np.sqrt(2*np.pi))**(-1)*np.exp(-((np.log(lin_slip)-MAP[i])**2)/(2*diag_std[i]**2))
                log_norms_new[i,:]=lognorm.pdf(lol_slip,diag_std[i],scale=np.exp(MAP[i]))
                integrals.append(trapz(log_norms[i,:],x=lin_slip))
                CIs[i],idx1[i],idx2[i]=self.cred_interval(log_norms[i,:],
                                                          MAP[i],
                                                          diag_std[i],
                                                          lin_slip)
                CIs_new[i]=lognorm.ppf(0.84,diag_std[i],scale=np.exp(MAP[i]))-lognorm.ppf(0.16,diag_std[i],scale=np.exp(MAP[i]))
        
      
       
        return CIs, CIs_new
        


       
    def cred_interval(self,lognormal,MAP,std,slip_vector):
        ## calculate 95% credible interval
        lim1=2.5/100
        lim2=97.5/100
        ## calculate 67# credible interval
        lim1=16/100
        lim2=83/100
        lim1=0.15
        lim2=0.85
        x2_5=np.exp(np.sqrt(2)*std*erfinv(2*lim1-1)+MAP)
        x97_5=np.exp(np.sqrt(2)*std*erfinv(2*lim2-1)+MAP)
        #print('xD')
        #print(MAP,std,x2_5,x97_5)
        idx1=find_nearest(x2_5,slip_vector)
        idx2=find_nearest(x97_5,slip_vector)
        #CI=slip_vector[idx2]-slip_vector[idx1]
        CI=x97_5-x2_5
        return CI,idx1,idx2

    
  
    def data_inversion(self,data='GNSS',cdg='data',cdi='nocorr',noise=0.1,CD_insar=3.0,gama=1.0):
        ## use uncertainties when doing the final inversion
        
        
        if data=='JOINT':
            Lc=self.Lc
            Ue_load = [elem.get('UE') for elem in self.data_estaciones]
            Un_load = [elem.get('UN') for elem in self.data_estaciones]
            Uz_load = [elem.get('UU') for elem in self.data_estaciones]
            data_gnss=np.hstack((Ue_load,Un_load,Uz_load))

            Stde_load = np.array([float(elem.get('SE')) for elem in self.data_estaciones])                
            Stdn_load = np.array([float(elem.get('SN')) for elem in self.data_estaciones])
            Stdz_load = np.array([float(elem.get('SU')) for elem in self.data_estaciones])
            covar_vector=np.hstack((Stde_load,Stdn_load,Stdz_load))
            CD_gnss=np.diag(covar_vector**(2))

            Ulos_load=[elem.get('LOS') for elem in self.data_estaciones_insar]
            data_insar=np.array(Ulos_load)
            print('q',CD_insar.shape)
            std=0.1
            CD_insar=std**2*CD_insar
            CD=block_diag((CD_gnss,gama*CD_insar)).toarray()
            data=np.hstack((data_gnss,data_insar))
            return data,CD,CD_gnss,CD_insar

            
      
        
    