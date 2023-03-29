# -*- coding: utf-8 -*-

import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colorbar as cbar

import numpy as np
import math
import os
import scipy as scp
import sympy
from scipy.linalg import cholesky
from scipy.stats import lognorm,norm
from scipy.special import erf,erfinv,kv,gamma
from scipy.integrate import simps,quad,nquad,dblquad,trapz
from scipy.interpolate import griddata,interp1d
from scipy.optimize import nnls,minimize,minimize_scalar,fmin,basinhopping,Bounds
from scipy.sparse import block_diag
from datetime import datetime
from matplotlib import colors
import time    
from fractions import Fraction
from functools import reduce
import timeit
import csv
import shutil
import sys
import random
import string

"""
Class for constructing okada green functions according to Slab2.0 geometry (Hayes et al., 2018).
Suited for South-Central chile NZ-SA plate boundary. Not tested for other enviroments.

Updated from previous work of UdeC undergraduates Felipe Vera (2016) and Leonardo Aguirre (2018).
"""
class Seismic_Cycle:

    def __init__(self, *args, **kwargs):
        
        ## hiperparametros que usamos para la inversion por optimizacion 
        self.lmbd1=kwargs.get('lambda1',0.02)
        self.lmbd2=kwargs.get('lambda2',0.02)
        self.lmbd3=kwargs.get('lambda3',0.02)
        self.lmbd4=kwargs.get('lambda4',0.02)

        ## una ID del proceso para diferenciarla de las demas.
        self.id_proceso = kwargs.get('ID', '/Output_1')
        ## elegir entre coseismic, interseismic o postseismic
        self.proceso = kwargs.get('proceso','coseismic')
        ## elegir entre Single o Double
        self.interface = kwargs.get('interface','double')
        # Elegir entre Directo o Inversion o Green_Functions
        self.output = kwargs.get('output','direct')


       ## create folder hierarchy
        self.crear_arquitectura=kwargs.get('create_folders',True)
        if self.crear_arquitectura:
            try:
                os.mkdir('Outputs'+ self.id_proceso)
            except FileExistsError:
                    print('la carpeta ya existe')

            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso)
            except FileExistsError:
                print('la carpeta ya existe')

            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface)
            except FileExistsError:
                print('la carpeta ya existe')

            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output)
            except FileExistsError:
                print('la carpeta ya existe')


            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf')
            except FileExistsError:
                print('la carpeta ya existe')

            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf_insar')
            except FileExistsError:
                print('la carpeta ya existe')


            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/figs')
            except FileExistsError:
                print('la carpeta ya existe')

            try:
                os.mkdir('Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt')
            except FileExistsError:
                print('la carpeta ya existe')

    
        # I dont use this variable on script    
        self.path_slips=kwargs.get('path_slips','Outputs')
        ##  Whether we create smooth matrix or not
        self.makeF = kwargs.get('smooth_matrix',False)
        ## Whether we create minima matriz or not
        self.makeM = kwargs.get('minima_matrix',True)
        ## Unused variable
        self.nombre_zona = kwargs.get('zona', 'Chile')
        ## define the limits in which data will be collected for creating slab geometry
        self.limites_zona = kwargs.get('limites_zona',(-18, -50))
        ## this should be always 3
        self.n_dim = kwargs.get('dimensiones', 3)
        ## the initial dip angle assumed for the fault plane
        self.dip_plano_inicial = kwargs.get('dip_inicial', np.radians(14.))
        ## initial depth assumed for fault plane
        self.depth_plano_inicial = kwargs.get('depth_inicial', 50000.0)
        ### width of the fault along dip (a.k.a W in okada formulation)
        self.ancho_placa = kwargs.get('ancho_placa',400000.0)
        ancho_placa_max=self.depth_plano_inicial/np.sin(self.dip_plano_inicial)
        ## proyeccion en la horizontal del ancho de la placa
        self.dy=kwargs.get('dy',30000.0)
        ## largo subfalla a lo largo del strike
        self.dx=kwargs.get('dx',4000)
        
       
        self.check_geometry=kwargs.get('check_geometry',True)
        ## this delta is used for building subfaults coordinates as needed in okada formulation
        self.delta_lat = kwargs.get('delta_lat', 0.08)
        ## input plate velocity in meters per year
        self.velocidad_placa = kwargs.get('vel_placa', 0.068)
        ## reference rake for fault slip assuming thrust faulting
        self.rake_referencia = kwargs.get('rake_referencia', 120)
        # this is plate thickness on m
        self.plate_thickness = kwargs.get('slab_thickness', 11000)
        ## length of the'fault along strike. Must intialize those variables
        self.L = None
        self.strike_aux_deg = None
        self.backstrike = None
        ## parameters for creating subfaults and interpolating
        self.numcols=50
        self.numrows=150
        ## date
        self.date = kwargs.get('date', datetime.utcnow().isoformat())
        ## Chile  case is always WE
        self.sentido_subduccion = kwargs.get('sentido', 'WE')
        ## whether we generate a boundary condition on trench
        ## I think this is not needed in the new way to code this
        self.condicion_borde_fosa = kwargs.get('condicion_borde_fosa', False)
        ## FILES WHERE WE GET SLAB PARAMETERS ACCORDING TO SLAB 1.0 - 2.0
        self.file_fosa = kwargs.get(
            'file_fosa',
            "./DatosBase/slab2/fosa_chile.csv")
        ## SLAB 2
        self.file_fosa = kwargs.get(
            'file_dip',
            "./DatosBase/slab2/slab2_perimeter_chile.csv")

        self.file_profundidad = kwargs.get(
            'file_profundidad',
            "./DatosBase/slab2/slab2_depth_chile.csv")
        self.file_strike = kwargs.get(
            'file_strike',
            "./DatosBase/slab2/slab2_strike_chile.csv")
        self.file_dip = kwargs.get(
            'file_dip',
            "./DatosBase/slab2/slab2_dip_chile.csv")
        # self.file_magica = kwargs.get(
        #     'file_dip',
        #     "./DatosBase/slab2/slab2_perimeter_chile.csv")
        ## not needed at the moment
        self.file_thick = kwargs.get('file_thick', "./DatosBase/slab2/slab2_thick_chile.csv" )
        ## griding of faults (# along strike, # along dip)
        self.grid_size = kwargs.get('grid_size', (75,5))
        
        self.bypass = kwargs.get('bypass',False)
        if not self.bypass:
            ## if bypass is false, then we do long process and 
            ##  fault data is loaded 
            self.data_falla = self.load_data_falla()
        self.synthetic_test=kwargs.get('synthetic_test', True)
        self.load_incertezas=kwargs.get('load_incertezas',True)
        ## whether we will generate strikeslip and dipslip separate sols.
        ## in other words if we will fix rake or not 
        self.split_G = kwargs.get('split_G', False)
        
        self.data_insar=kwargs.get('insar',False)
        self.Lc=kwargs.get('Lc',10)
        if self.bypass:
            ## if bypass is true, then a bypass is performed and 
            ## fault geometry and green functions are loaded (geometry is not generated),
            ## and process takes less time
            self.green_functions,self.coord_fallas=self.load_greenfunctions()

        ## which fault planes will be constructed
        self.falla_AB=kwargs.get('plano_AB', False)
        self.falla_CD=kwargs.get('plano_CD', False)
        self.falla_E=kwargs.get('plano_E',  False)

        
        ## synthetic slip generated data (from moreno et al.)
        self.slip_sintetico={}
        ## inverted slip generated
        self.slip_invertido={}
        ## Fault planes information (endpoints, number of faults, etc)
        self.planos_falla = {}
        ## In depth info of fault plane
        self.planos_falla_obj = {}
        ## Information of fault griding 
        self.subfallas_model_dict={}
        ## Information of fault CD griding
        self.subfallas_model_dict_CD={}
        ## Information of fault E griding
        self.subfallas_model_dict_E={}




        ## name of GNSS data used 
        self.name_file=kwargs.get('data_observada','./Fuentes/estaciones_cosismico.csv')
        ## load GNSS data
        self.data_estaciones=self.load_GNSS()
        
        ## CREATE INSAR DATA FILE
        self.name_file_insar=kwargs.get('data_observada_insar','./Fuentes/estaciones_cosismico.csv')
        if self.data_insar:
            self.data_estaciones_insar=self.load_insar()
       
        ## name of output folders and files
        if self.proceso == 'coseismic':
            label='disp'
        if self.proceso == 'interseismic':
            label='velo'
        self.desplaz_output='Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt'+self.id_proceso+'_'+label+'_generated.txt'
        self.desplaz_output_insar='Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt'+self.id_proceso+'_'+label+'_INSAR_generated.txt'
        self.lower_interface_output='Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt/'+self.proceso.lower()+'_slip_lower_interface.txt'
        self.upper_interface_output='Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt/'+self.proceso.lower()+'_slip_upper_interface.txt'
        self.figuras_output='Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/figs/'
        self.inversion_params = 'Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt/inv_params.txt'
        ## Green Functions Output. Unused
        self.gf_upper_output = 'Outputs'+self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/txt/'

        ## generated synthetic data by a DIRECT PROBLEM
        self.data_sintetica={}
        ## outputs of inversion. Apparently unused
        self.outputs_inv={}
        ##inverted velocities
        self.inverted_data={}

    def generar_estaciones(self,nx,ny):
        latmin=self.limites_zona[0]+0.2
        latmax=self.limites_zona[1]-0.2
        lonmin=-73
        lonmax=-72
        sta_x=np.linspace(latmin,latmax,nx)
        sta_y=np.linspace(lonmin,lonmax,ny)
        lon_sta=[]
        lat_sta=[]
        name_sta=[]
        UE=np.ones(nx*ny)
        UN=np.ones(nx*ny)
        UZ=np.ones(nx*ny)
        #xx,yy=np.meshgrid(estaciones_x,estaciones_y)
        for i in range(nx):
            for j in range(ny):
                lon_sta.append(sta_y[j])
                lat_sta.append(sta_x[i])
                name=get_random_alphaNumeric_string()
                name_sta.append(name)

        with open('estaciones_sint.txt','w') as csvfile:
                writer=csv.writer(csvfile,delimiter=';')
                writer.writerow(['Station','Longitud','Latitud','UE','UN','UZ'])
                for row in zip(name_sta,lon_sta,lat_sta,UE,UN,UZ):
                    writer.writerow(row)
        return lon_sta,lat_sta,name_sta

    def load_data_falla(self):
        ## this function cuts the main failes of dip, strike, depth and the trench latlon in
        ## the interval given bit zhone limits defined in init function

        data_type = {'Latitud': float,
                     'Longitud': float,
                     'Strike': float,
                     'Dip': float,
                     'Profundidad': float,
                     'Thickness': float}
        data_profundidad = cut_file(
            self.file_profundidad,
            data_type,
            self.limites_zona,
            'Latitud')
        data_fosa = cut_file(
            self.file_fosa,
            data_type,
            self.limites_zona,
            'Latitud')
        data_strike = cut_file(
            self.file_strike,
            data_type,
            self.limites_zona,
            'Latitud')
        data_dip = cut_file(
            self.file_dip,
            data_type,
            self.limites_zona,
            'Latitud')


        # data_thick = cut_file(
        #     self.file_thick,
        #     data_type,
        #     self.limites_zona,
        #     'Latitud')
    
        data_falla = dict(
            fosa=data_fosa,
            profundidad=data_profundidad,
            strike=data_strike,
            dip=data_dip)
            #thickness=data_thick)

        return(data_falla)

   
    
    

    def load_GNSS(self,  stations=[], load_incertezas=False):
        ##  this function loads the observed data related to the analyzed seismic period. the limits 
        ##  are required only if we have data in a greater space than we want to use. This data will be
        #   used in the inversion process
        
        if not self.load_incertezas:
            data_type = {
                         'Latitud': float,
                         'Longitud': float,
                         'UE': lambda x: float(x)/1000,
                         'UN': lambda x: float(x)/1000,
                         'UU': lambda x: float(x)/1000
                        }
        if self.load_incertezas:
            if self. proceso=='coseismic':
                data_type = {
                             'Latitud': float,
                             'Longitud': float,
                             'UE': lambda x: truncate(float(x)/1000,5),
                             'UN': lambda x: truncate(float(x)/1000,5),
                             'UU': lambda x: truncate(float(x)/1000,5),
                             'SE': lambda x: truncate(float(x)/1000,5),
                             'SN': lambda x: truncate(float(x)/1000,5),
                             'SU': lambda x: truncate(float(x)/1000,5),
                            }
            if self. proceso=='interseismic':
                data_type = {
                             'Latitud': float,
                             'Longitud': float,
                             'UE': lambda x: truncate(float(x)/1000,5),
                             'UN': lambda x: truncate(float(x)/1000,5),
                             'SE': lambda x: truncate(float(x)/1000,5),
                             'SN': lambda x: truncate(float(x)/1000,5),
                            }
        archivo = self.name_file
        data_disp = cut_file(archivo, data_type)
        data_disp_set = data_disp
    
        if stations:
            data_disp_set = [data for data in data_disp
                                  if data.get('Station') in stations]
       
        return data_disp_set

    def load_insar(self):
        data_type = {
                    'Latitud': float,
                    'Longitud': float,
                    'LOS': lambda x: float(x)/1000,
                    'SLO': lambda x: float(x)/1000,
                    'EE':  lambda x: float(x),
                    'NN':  lambda x: float(x),   
                    'UU':  lambda x: float(x),
                    'TOPO':  lambda x: float(x)}
        archivo=self.name_file_insar

        data_disp=cut_file(archivo,data_type)
        return data_disp 
 

  

    def set_map_params(self, **kwargs):
        """ this function sets the Basemap params of build_map function"""
         ## escribir info. adicional
        ## this function sets the Basemap params of build_map function
        self.map_params = dict(
            projection='merc',
            llcrnrlat=kwargs.get('latmin'),
            urcrnrlat=kwargs.get('latmax'),
            llcrnrlon=kwargs.get('lonmin'),
            urcrnrlon=kwargs.get('lonmax'),
            lat_ts=(kwargs.get('latmin')+kwargs.get('latmax'))/2,
            resolution='h',)
        

    
    

    
        
   
    def data_wo_map(self,**kwargs):
        data_class=kwargs.get('data_class','hori')
        gnss_sint=kwargs.get('gnss_sint',None)
        los_sint=kwargs.get('los_sint',None)
        both_arrows=kwargs.get('both_arrows',False)
        fallas=kwargs.get('fallas',None)
        slip=kwargs.get('slip',None)

        nx=self.grid_size[0]
        ny=self.grid_size[1]
        fig,ax=plt.subplots()
        

        data_type = {'Latitud': float,
                     'Longitud': float,
                     'Strike': float,
                     'Dip': float,
                     'Profundidad': float,
                     'Thickness': float}

        data_fosa=cut_file(self.file_fosa,data_type,(-10,-40))
        fosa_lat = [elem.get('Latitud') for elem in data_fosa]
        fosa_lon = [elem.get('Longitud') for elem in data_fosa]
        ax.plot(fosa_lon,fosa_lat,'--r',linewidth=3)

        if self.data_type=='GNSS':
            ## observed data
            lon = [elem.get('Longitud') for elem in self.data_estaciones]
            lat = [elem.get('Latitud') for elem in self.data_estaciones]
            sta = [elem.get('Station') for elem in self.data_estaciones]

            UE = np.array([elem.get('UE') for elem in self.data_estaciones])
            UN = np.array([elem.get('UN') for elem in self.data_estaciones])
            UZ = np.array([elem.get('UU') for elem in self.data_estaciones])

            SE = np.array([elem.get('SE') for elem in self.data_estaciones])
            SN = np.array([elem.get('SN') for elem in self.data_estaciones])
            SZ = np.array([elem.get('SU') for elem in self.data_estaciones])
            for i in range(len(sta)):
                ax.text(lon[i], lat[i], sta[i],fontsize=10)
                ax.plot(lon,lat,'k*')

            ## synthetic_data  
            UE_sint=gnss_sint[0].flatten()
            UN_sint=gnss_sint[1].flatten()
            UZ_sint=gnss_sint[2].flatten()

            ## residuales o ambas flechas
            if data_class=='hori':
                scale=20.0
                length_vect=1.0
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.array(UE),np.array(UN),color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.array(UE_sint),np.array(UN_sint),color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()

                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.abs(UE-UE_sint),np.abs(UN-UN_sint),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
                
            if data_class=='vert':
                scale=6.0
                length_vect=0.5
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),UZ,color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),UZ_sint,color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()
                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),np.abs(UZ-UZ_sint),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
            

            ax.quiverkey(Q,0.2,0.55,length_vect,r'GPS observations')
            ax.quiverkey(Q,0.8,0.2,length_vect,str(length_vect)+r'$[m]$ displacement')
            ax.set_xlim([-79,-68])
            ax.set_ylim([-42,-31])

        if self.data_type=='INSAR':
                lon = [elem.get('Longitud') for elem in self.data_estaciones_insar]
                lat = [elem.get('Latitud') for elem in self.data_estaciones_insar]
                los = np.array([elem.get('LOS') for elem in self.data_estaciones_insar])
                if data_class=='insar_asc':
                    los=los[:len(los_sint)]
                    los_s=los_sint
                if data_class=='insar_des':
                    los=los[(len(los)-len(los_sint)):]
                    los_s=los_sint
                else:
                    los_s=los_sint

                scale=20.0
                length_vect=0.5
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(los)),los,color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.zeros(len(los_s)),los_s,color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()
                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(los)),np.abs(los-los_s),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
                ax.quiverkey(Q,0.2,0.55,length_vect,r'GPS observations')
                ax.quiverkey(Q,0.8,0.2,length_vect,str(length_vect)+r'$[m]$ displacement')
                ax.set_xlim([-79,-68])
                ax.set_ylim([-42,-31])
                ax.set_title(data_class)
        if self.data_type=='JOINT':
            if data_class=='gnss_hori':
                lon = [elem.get('Longitud') for elem in self.data_estaciones]
                lat = [elem.get('Latitud') for elem in self.data_estaciones]
                sta = [elem.get('Station') for elem in self.data_estaciones]

                UE = np.array([elem.get('UE') for elem in self.data_estaciones])
                UN = np.array([elem.get('UN') for elem in self.data_estaciones])

                SE = np.array([elem.get('SE') for elem in self.data_estaciones])
                SN = np.array([elem.get('SN') for elem in self.data_estaciones])
                
                ## synthetic_data  
                UE_sint=gnss_sint[0].flatten()
                UN_sint=gnss_sint[1].flatten()
                for i in range(len(sta)):
                    ax.text(lon[i], lat[i], sta[i],fontsize=10)
                    ax.plot(lon,lat,'k*')

                scale=20.0
                length_vect=1.0
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.array(UE),np.array(UN),color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.array(UE_sint),np.array(UN_sint),color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()

                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.abs(UE-UE_sint),np.abs(UN-UN_sint),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
                ax.quiverkey(Q,0.2,0.55,length_vect,r'GPS observations')
                ax.quiverkey(Q,0.8,0.2,length_vect,str(length_vect)+r'$[m]$ displacement')
                ax.set_xlim([-79,-68])
                ax.set_ylim([-42,-31])
            if data_class=='gnss_vert':
                lon = [elem.get('Longitud') for elem in self.data_estaciones]
                lat = [elem.get('Latitud') for elem in self.data_estaciones]
                sta = [elem.get('Station') for elem in self.data_estaciones]
                UZ = np.array([elem.get('UU') for elem in self.data_estaciones])
                UZ_sint=gnss_sint[2].flatten()

                scale=6.0
                length_vect=0.5
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),UZ,color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),UZ_sint,color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()
                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(UZ)),np.abs(UZ-UZ_sint),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
            

                ax.quiverkey(Q,0.2,0.55,length_vect,r'GPS observations')
                ax.quiverkey(Q,0.8,0.2,length_vect,str(length_vect)+r'$[m]$ displacement')
                ax.set_xlim([-79,-68])
                ax.set_ylim([-42,-31])
            

            ## residuales o ambas flechas
            if data_class=='insar':
                lon = [elem.get('Longitud') for elem in self.data_estaciones_insar]
                lat = [elem.get('Latitud') for elem in self.data_estaciones_insar]
                los = np.array([elem.get('LOS') for elem in self.data_estaciones_insar])
                los_s=los_sint.flatten()

                scale=20.0
                length_vect=0.5
              
                if both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(los)),los,color='r',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Obs')
                    Q=ax.quiver(lon,lat,np.zeros(len(los_s)),los_s,color='k',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26,
                                              label='Inv')
                    plt.legend()
                if not both_arrows:
                    Q=ax.quiver(lon,lat,np.zeros(len(los)),np.abs(los-los_s),color='b',scale=scale,
                                              width=0.00250,
                                              linewidth=0.5,
                                              headwidth=4.,
                                              zorder=26)
                ax.quiverkey(Q,0.2,0.55,length_vect,r'GPS observations')
                ax.quiverkey(Q,0.8,0.2,length_vect,str(length_vect)+r'$[m]$ displacement')
                ax.set_xlim([-79,-68])
                ax.set_ylim([-42,-31])
                ax.set_title(data_class)

        if fallas:
                lon_fallas_ab=fallas[0]
                lat_fallas_ab=fallas[1]
                k=0
                for lonx,laty in zip(lon_fallas_ab,lat_fallas_ab):
                    ax.plot(lonx,laty,color='k',  linestyle = '-', linewidth = 1,zorder=21)
                    if slip is not None:
                        normal = plt.Normalize(slip.min(), slip.max())
                        cmap = plt.cm.Reds(normal(slip))
                        ax.fill_between(lonx,laty,facecolor=cmap[k])
                        k+=1
        if slip is not None:
                cax,aux=cbar.make_axes(ax)
                cb2=cbar.ColorbarBase(cax,cmap=plt.cm.Reds,norm=normal)



                    


    
    
    ################################################################################
    
    def proj_mesh2okada(self,lat_falla,lon_falla,strike,insar=False): 
        ## this function obtains the (xi,yi) coordinates in the okada reference frame for given station data in the 
        ## geographic reference frame
        ## lat_falla : subfault latitude
        ## lon_falla : subfault longitude
        ## strike: strike of the subfault  """ 

        # OUTPUT: coordenadas xi,yi proyectadas en los ejes de Okada
        #OBS FELIPE: La funcion determina las coordenadas okada (X Y) EN SUPERFICIE de un un punto
        #de observacion a partir de sus coordenadas geograficas.        
        lon_estaciones = [elem.get('Longitud') for elem in self.data_estaciones]
        lat_estaciones = [elem.get('Latitud') for elem in self.data_estaciones]
        #name_estaciones=[elem.get('Station') for elem in self.data_estaciones]

        if insar:
            lon_estaciones = [elem.get('Longitud') for elem in self.data_estaciones_insar]
            lat_estaciones = [elem.get('Latitud') for elem in self.data_estaciones_insar]
            #name_estaciones=[elem.get('Station') for elem in self.data_estaciones_melnick]
        dist_lat=[]
        dist_lon=[]
        az_m=[]
        l_m=[]
        baz_m=[]
        bandera = False
       
        for (lat_sta,lon_sta) in zip(lat_estaciones,lon_estaciones):
          ## esta funcion calcula la distancia entre dos puntos geograficos
           l,az,baz=vinc_dist(lat_falla,lon_falla,lat_sta,lon_sta)
           
           az_m.append(az)
           l_m.append(l)
           baz_m.append(baz)
           if bandera:
               if az <= 180:
                    dist_lat.append(l*np.cos(np.radians(az)))
                    dist_lon.append(l*np.sin(np.radians(az)))
               if az > 180 :
                    dist_lat.append(l*np.cos(np.radians(360-az)))
                    dist_lon.append(l*np.sin(np.radians(360-az)))

           if not bandera:
               dist_lat.append(l*np.cos(np.radians(az)))
               dist_lon.append(l*np.sin(np.radians(az)))


       
        dist_lat_arr=np.array(dist_lat)
        dist_lon_arr=np.array(dist_lon)
        xi = dist_lat_arr*np.cos(strike) + dist_lon_arr*np.sin(strike)
        yi = dist_lat_arr*np.sin(strike) - dist_lon_arr*np.cos(strike)
        
        return xi,yi
   

    def crear_subfallas(self):
        """
        this function creates the subfaults geometry with depth and dip mapped from Slab 2.0 (Hayes et al 2018)
        strike is fixed to 18 degrees in the study area for simplicity.
        """
        self.strike_ref=18.0
        nx=self.grid_size[0]
        ny=self.grid_size[1]
        fosa = self.data_falla.get('fosa')
        ## load trench points. sort from lower to greater
        fosa_lat = [elem.get('Latitud') for elem in fosa]
        fosa_lon = [elem.get('Longitud') for elem in fosa]
        idx=np.argsort(fosa_lat)
        fosa_lat=np.array(fosa_lat)[idx]
        fosa_lon=np.array(fosa_lon)[idx]

        pb=6
        lat_tr_final,lon_tr_final=self.interpolar_trench(fosa_lat,fosa_lon,pb=pb)
       
        L,aux1,aux2=vinc_dist(lat_tr_final[0],lon_tr_final[0],lat_tr_final[-1],lon_tr_final[-1])
        self.dx=L/nx
        dx=self.dx
        dx_arr=np.linspace(0,L,nx+1)[1:-1]
        dist=np.zeros(len(lat_tr_final)-1)

        ##proyeccion en la horizontal del ancho de la subfalla 
        dy=self.dy
        interpolar=True
        if nx>1 and interpolar:
            for i in range(len(lat_tr_final)-1):
                dist[i],aux,aux=vinc_dist(lat_tr_final[0],
                                       lon_tr_final[0],
                                       lat_tr_final[i+1],
                                       lon_tr_final[i+1])
            idxs=np.zeros(nx-1)
            for j in range(len(dx_arr)):
                idxs[j]=find_nearest(dist,dx_arr[j])
            idxs=idxs.astype(np.int64)
            
       
            ## puntos de la fosa desde donde se construiran los nodos
            lat_ori=np.hstack((lat_tr_final[0],lat_tr_final[idxs],lat_tr_final[-1]))
            lon_ori=np.hstack((lon_tr_final[0],lon_tr_final[idxs],lon_tr_final[-1]))
        if nx>1 and not interpolar:
            lat_ori=lat_tr_final
            lon_ori=lon_tr_final
        if nx==1:
            lat_ori=np.array([lat_tr_final[0],lat_tr_final[-1]])
            lon_ori=np.array([lon_tr_final[0],lat_tr_final[-1]])

        depth_fosa=self.get_points_slab2(lat_ori,lon_ori,key='Profundidad')
        depth_fosa=depth_fosa[:-1]
        depth_fosa=(depth_fosa*np.ones((ny,nx))).reshape((nx*ny,))

        
        lat_nodo,lon_nodo=self.improve_geometry(lat_ori,lon_ori,dy,fix_strike=True)
        


     
      
        ## crear vertices
        strike_ref=self.strike_ref*np.ones(nx*ny)
        strike=strike_ref
    
        #vert_lat,vert_lon,lat_cent,lon_cent=self.crear_vertices(lat_nodo,lon_nodo,dx,dy,strike,fix_strike=False)
        vert_lat,vert_lon,lat_cent,lon_cent=self.crear_vertices(lat_nodo,lon_nodo,dx,dy,strike,fix_strike=True)
        ## vertices
        ## vertices inferior derecho
        lat1=[x[0] for x in vert_lat]
        lon1=[x[0] for x in vert_lon]
        ## vertices superior derecho
        lat2=[x[1] for x in vert_lat]
        lon2=[x[1] for x in vert_lon]
        ## vertices superior izquierdo
        lat3=[x[2] for x in vert_lat]
        lon3=[x[2] for x in vert_lon]
        ## vertices inferior izquierdo
        lat4=[x[3] for x in vert_lat]
        lon4=[x[3] for x in vert_lon]
        ## depth en km. No usado
        ## obtener profundidad a partir de los vertices inferiores
        depth1=self.get_points_slab2(lat1,lon1,key='Profundidad')
        depth2=self.get_points_slab2(lat2,lon2,key='Profundidad')
        depth3=self.get_points_slab2(lat3,lon3,key='Profundidad')
        depth4=self.get_points_slab2(lat4,lon4,key='Profundidad')

        depth=(depth1+depth2)/2
        #depth=depth
        ## obtener dip a partir de los vertices superiores
        dip1=self.get_points_slab2(lat3,lon3,key='Dip')
        dip2=self.get_points_slab2(lat4,lon4,key='Dip')
        dip=(dip1+dip2)/2
       
        width1=dy/np.cos(np.radians(dip))
        phi_placa_rad = np.radians(strike + (360-(self.rake_referencia+180)))
        rake_inv_rad  = []
        ## rake for normal fault
        rake_norm_rad = []
        for index, elem in enumerate(strike):
            # Se establece rake en subfallas emplazadas en la
            # interfase superior con un movimiento relativo de tipo inverso
            try:
                rake = get_rake(
                                phi_placa_rad[index],
                                np.radians(elem),
                                np.radians(dip[index])
                                )
                #rake=np.radians(90)
            except IndexError:
                rake = get_rake(
                                phi_placa_rad[index],
                                np.radians(elem),
                                np.radians(dip)
                                )
            rake_inv_rad.append(rake)
            # Se establece rake en subfallas emplazadas en la interfase
            # inferior con un movimiento relativo de tipo normal
            rake_norm_rad.append(rake_inv_rad[index]+np.pi)


        depth_normal= depth+ self.plate_thickness/1000/np.cos(np.radians(dip))

        self.subfallas_model_dict.update({
                                        'lon_central':lon_cent,
                                        'lat_central':lat_cent,
                                        'lon_vertices':vert_lon,
                                        'lat_vertices':vert_lat,
                                        'length_model':dx,
                                        'width_model':width1,
                                        'dip_model':np.radians(dip),
                                        'strike_model':strike,
                                        'depth_model':depth,
                                        'rake_inv_rad':rake_inv_rad,
                                        'rake_norm_rad':rake_norm_rad,
                                        'phi_placa_rad':phi_placa_rad,
                                        'depth_normal':depth_normal})

       
        return depth,width1,depth_normal,dx,strike,dip,vert_lon,vert_lat
   def get_points_slab2(self,lat_nodo,lon_nodo,key='Profundidad'):
    """ 
    this function reads SLAB2.0 data (Hayes et al., 2018) to map strike, dip and depth data to the subfaults
    """
        if key not in ['Profundidad','Strike','Dip','Thickness']:
            print('La llave ha sido mal ingresada')
        lats=[ elem.get('Latitud') for elem in self.data_falla.get(key.lower())]
        lons=[ elem.get('Longitud') for elem in self.data_falla.get(key.lower())]
        data=[ elem.get(key) for elem in self.data_falla.get(key.lower())]
        xi,yi=np.meshgrid(lat_nodo,lon_nodo)
        zi=griddata((np.array(lats),np.array(lons)),
                     np.array(data),(xi,yi),
                     method='nearest')
        
        ## los puntos que queremos estan en la diagonal
        if key in['Profundidad']:
            return np.diag(-zi)
        if key in ['Thickness','Dip']:
            return np.diag(zi)
        if key in ['Strike']:
            strike=np.diag(zi)
            strike_new=np.zeros(len(lat_nodo))
            for i in range(len(strike)):
                if strike[i]>180:
                   strike_new[i]=strike[i]-360
                else:
                    strike_new[i]=strike[i]
            return strike_new

    def interpolar_trench(self,fosa_lat,fosa_lon,pb=2):
        if pb==1:
            print('ERROR: var pb no puede ser igual a 1')
        n=len(fosa_lon)
        k=0
        new_lats=np.zeros(pb*(len(fosa_lon)-1))
        new_lons=np.zeros(pb*(len(fosa_lon)-1))
        

        for i in range(n-1):

            dist,az,baz=vinc_dist(fosa_lat[i],fosa_lon[i],fosa_lat[i+1],fosa_lon[i+1])
            for j in range(pb):
                new_lats[k],new_lons[k],aux=vinc_pt(fosa_lat[i],fosa_lon[i],az,dist*(j+10**(-12))/pb)
                k+=1
    
        return  new_lats,new_lons

   
    def improve_geometry(self,lat_ori,lon_ori,dy,fix_strike=False):
        nx=self.grid_size[0]
        ny=self.grid_size[1]
        
        strike=np.zeros(nx*ny)
        if fix_strike:
            strike=self.strike_ref*np.ones(nx*ny)

        STRIKES=[]
        DIFFS_LATS=[]
        DIFFS_LONS=[]
        LATS_NODOS=[]
        LONS_NODOS=[]
        k=0
        while k < 10:
            lat_nodo=np.zeros(nx*ny)
            lon_nodo=np.zeros(nx*ny)
            if k==0:
                ## Cuando k =0, obtenemos el strike[i] de vinc_dist, lo que sera nuestro
                ## strike_guess para cada nodo de la fosa
                    ## al nodo i+1
                for i in range(nx*ny):
                    
                    if i < nx:
                        ## en este caso solo hay una subfalla a lo largo del dip
                        if not fix_strike:
                            dist,strike[i],aux=vinc_dist(lat_ori[i],lon_ori[i],lat_ori[i+1],lon_ori[i+1])

                        ## mover el nodo en direccion dy una distancia dy
                        lat_nodo[i],lon_nodo[i],aux=vinc_pt(lat_ori[i],lon_ori[i],strike[i]+90,dy)

                       
                    if i >= nx:
                        ## cuando hay mas de una falla a lo largo del dip
                        strike[i]=strike[i-nx]
                        lat_nodo[i],lon_nodo[i],aux=vinc_pt(lat_nodo[i-nx],lon_nodo[i-nx],strike[i-nx]+90,dy)
                
            if k>0:
                ##  cuando k>0, el strike usado para consturir los nodos es el strike dedujido
                ##  de slab2.0 usando k-1
                for i in range(nx*ny):
                ## calcular el strike esperado desde el nodo i de la fosa
                ## al nodo i+1
                    if i < nx:
                    ## mover el nodo en direccion dy una distancia dy
                        lat_nodo[i],lon_nodo[i],aux=vinc_pt(lat_ori[i],lon_ori[i],strike[i]+90,dy)
                   
                    if i >= nx:
                        lat_nodo[i],lon_nodo[i],aux=vinc_pt(lat_nodo[i-nx],lon_nodo[i-nx],strike[i-nx]+90,dy)
            if not fix_strike:
                strike=self.get_points_slab2(lat_nodo,lon_nodo,key='Strike')
            if fix_strike:
                strike=self.strike_ref*np.ones(nx*ny)
            #STRIKES.append(strike)
            LATS_NODOS.append(lat_nodo)
            LONS_NODOS.append(lon_nodo)
            if k>0:
                diff_lat=np.sum(np.abs(LATS_NODOS[k]-LATS_NODOS[k-1]))
                diff_lon=np.sum(np.abs(LONS_NODOS[k]-LONS_NODOS[k-1]))

                DIFFS_LATS.append(diff_lat)
                DIFFS_LONS.append(diff_lon)
                if diff_lat < thr and diff_lon < thr:
                
                    return lat_nodo,lon_nodo
                    break
            thr=10**(-6)

                
           
            k+=1
        return lat_nodo,lon_nodo
    
    
       

    def crear_vertices(self,lat_nodo,lon_nodo,dx,dy,strike,fix_strike=False):
        vert_lat=len(lat_nodo)*[5*[0]]
        vert_lon=len(lat_nodo)*[5*[0]]
        #Se evaluaran todas las subfallas.
        FIL=self.grid_size[0]*self.grid_size[1]
        if fix_strike:
            strike=self.strike_ref*np.ones(FIL)

        #Cada subfalla tiene cinco vertices a dibujar (el ultimo coincide con el primero)
        COL=5   
        
        #Se declaran variables para almacenar vertices de todas las subfallas.
        vert_lat    = []
        vert_lon    = []
        #Se crean matrices de ceros para rellenar con los vertices de todas las subfallas.
        for i in range(FIL):                #Numero de filas de la matriz
            vert_lon.append([0]*COL)    #Numero de columnas de la matriz, se inicializa con ceros
            vert_lat.append([0]*COL)
        lat_cent=np.zeros(FIL)
        lon_cent=np.zeros(FIL)

        for i in range(len(lat_nodo)):
                #vertice 0 es lat_nodo
                vert_lat[i][0]=lat_nodo[i]
                vert_lon[i][0]=lon_nodo[i]
                ## vertice 1
                vert_lat[i][1],vert_lon[i][1],aux=vinc_pt(lat_nodo[i],
                                                          lon_nodo[i],
                                                          strike[i],
                                                          dx)
                ## vertice 2
                vert_lat[i][2],vert_lon[i][2],aux=vinc_pt(vert_lat[i][1],
                                                          vert_lon[i][1],
                                                          strike[i]+270,
                                                          dy)
                ## vertice 3
                vert_lat[i][3],vert_lon[i][3],aux=vinc_pt(vert_lat[i][0],
                                                          vert_lon[i][0],
                                                          strike[i]+270,
                                                          dy)
                ## vertice 4
                vert_lat[i][4]=vert_lat[i][0]
                vert_lon[i][4]=vert_lon[i][0]
                ## puntos centrales
                aux1,aux2,aux3=vinc_pt(lat_nodo[i],
                                       lon_nodo[i],
                                       strike[i],
                                       dx/2)
                lat_cent[i],lon_cent[i],aux3=vinc_pt(aux1,aux2,strike[i]+270,dy/2)
        return vert_lat,vert_lon,lat_cent,lon_cent


    
    
    def interpolate_coupling(self):
        ## cargar el archivo de acoplamiento de metois et al. 2016
        ## dado que el archivo tiene solo un promedio del acopl. por latitud,
        ## generamos las longitudes para luego interpolar
        data=np.genfromtxt('DatosBase/coupling_maule.txt')
        coupling=data[:,1]
        lat=data[:,0]


        lon=np.arange(-73,-69,0.1)

        lon_coup=np.zeros(len(lon)*len(lat))
        lat_coup=np.zeros(len(lon)*len(lat))
        coup_deg=np.zeros(len(lon)*len(lat))
        k=0
        for i in range(len(lat)):
            for j in range(len(lon)):
                lon_coup[k]=lon[j]
                lat_coup[k]=lat[i]
                coup_deg[k]=coupling[i]
                k+=1
        
        # cargar los nodos donde queremos datos
        lon_central=np.array(self.subfallas_model_dict.get('lon_central'))
        lat_central=np.array(self.subfallas_model_dict.get('lat_central'))
        xi,yi=np.meshgrid(lon_central,lat_central)
        coup=griddata((lon_coup,lat_coup),coup_deg,
                    (lon_central,lat_central),method='nearest')
        nans=np.isnan(coup)
        coup[nans]=0
        ## convertir coupling a slip-rate
        ## velocidad de convergencia de las placas
        ## en metros/año
        vc=self.velocidad_placa 
        ## v_placa in m/yr
        v_placa=vc-(1-coup)*vc
        v_placa=vc*np.ones(len(v_placa))
       
        factor_inter_thrust=0.00
        factor_inter_normal=1.00

        self.slip_sintetico.update({'intersismico_thrust':factor_inter_thrust*v_placa})
        self.slip_sintetico.update({'intersismico_normal':factor_inter_normal*v_placa})
        self.slip_sintetico.update({'lon_falla':lon_central,
                                    'lat_falla':lat_central})
        return v_placa
        
    def interpolate_slip(self):
        data=np.genfromtxt('3_slip_moreno/slip_more_paper.txt')
        lon_central=np.array(self.subfallas_model_dict.get('lon_central'))
        lat_central=np.array(self.subfallas_model_dict.get('lat_central'))
        lon_slip=data[:,0]
        lat_slip=data[:,1]
        slip=data[:,2]
        xi,yi=np.meshgrid(lon_central,lat_central)
        slip=griddata((lon_slip,lat_slip),slip,
                    (lon_central,lat_central),method='nearest')
        nans=np.isnan(slip)
        slip[nans]=0
        factor_cos_thrust=1.0
        self.slip_sintetico.update({'cosismico_thrust':factor_cos_thrust*slip})
        factor_cos_normal=0.5
        self.slip_sintetico.update({'cosismico_normal':factor_cos_normal*slip})
        self.slip_sintetico.update({'lon_falla':lon_central,
                                    'lat_falla':lat_central})
        factor_inter_thrust=0.25
        factor_inter_normal=1.0
        ## scale coseismic slip by interseismic slip-rate
        slip_rate=slip/np.max(slip)*self.velocidad_placa
        self.slip_sintetico.update({'intersismico_thrust':factor_inter_thrust*slip_rate})
        self.slip_sintetico.update({'intersismico_normal':factor_inter_normal*slip_rate})
        nx=self.grid_size[0]
        ny=self.grid_size[1]
        slip2=slip
        slip3=slip2[::-1]
        slip=slip.reshape(nx,ny)
        return factor_cos_thrust*slip2,factor_cos_normal*slip3,np.vstack((lon_central,lat_central,factor_cos_thrust*slip2)).T

    


    def model_matrix_slab(self,falla='inversa',save_subfallas_methods=True, generar_subfallas = True,save_GF=True,model='slabmodel'):
        ## This function generates green functions A or B


        ### we generate the subfaults if its needed (it is not needed, for example, if we are building B after A.
        ### the geometry is the same, we only change the depth and the rake 
        if generar_subfallas:
            self.crear_subfallas()
        

        nx           = self.grid_size[0]
        ny           = self.grid_size[1]

        ### ALL THE DATA GENERATED BELOW IS USING SLAB 2.0 MODEL
        ## depth of the fault plane 
        profundidad_subfallas_inversa = self.subfallas_model_dict.get('depth_model')

        ## dip of the fault plane (will work for both planes)
        dip_subfallas_radianes  = self.subfallas_model_dict.get('dip_model')
    
        ## width of subfaults (same for all faults)
        width_subfallas= self.subfallas_model_dict.get('width_model')

        ## width of subfaults (same for all faults)
        length_subfallas= self.subfallas_model_dict.get('length_model')
        

        ## strike of the fault plane (will work for both planes)
        strike_subfallas_deg = self.subfallas_model_dict.get('strike_model')
        
        ## longitude of the vertex of fault plane (will work for both)
        lon_vertices_subfallas_slab = self.subfallas_model_dict.get('lon_vertices')

        ## latitude of the vertex of fault plane (will work for both)
        lat_vertices_subfallas_slab = self.subfallas_model_dict.get('lat_vertices')
        

        #we get the depth of the normal fault using a constant thickness of the slab 
        profundidad_subfallas_normal = profundidad_subfallas_inversa + \
                                        (self.plate_thickness/1000 / np.cos(dip_subfallas_radianes))

        phi_placa_rad = self.subfallas_model_dict.get('phi_placa_rad')


        ## rake for inverse fault
        rake_inv_rad  = self.subfallas_model_dict.get('rake_inv_rad')

        ## rake for normal fault
        rake_norm_rad = self.subfallas_model_dict.get('rake_norm_rad')

       

        if falla == 'inversa' and model == 'slabmodel':
            label='A'
            rake_usado = rake_inv_rad
            prof_usada = profundidad_subfallas_inversa
            

        if falla == 'normal' and model == 'savage':
            label = 'A'
            rake_usado = rake_norm_rad
            prof_usada = profundidad_subfallas_inversa
        if falla == 'normal' and model == 'slabmodel':
            label='B'
            rake_usado = rake_norm_rad
            prof_usada = profundidad_subfallas_normal
       
        ndim=self.n_dim
        
        subfalla=0
        

        for j_ny in range(ny):
            for i_nx in range(nx):
                ## in this case we use real width of the fault since we are using
                ## green functions
                if i_nx == 0  and j_ny == 0:
                    
                    
                    ddx,ddy= self.proj_mesh2okada(
                                                lat_vertices_subfallas_slab[subfalla][0],
                                                lon_vertices_subfallas_slab[subfalla][0],
                                                np.radians(strike_subfallas_deg[j_ny]),
                                                melnick=False)

                    try:
                        ve,vn,vz = desplaz_okada(
                                                 ddx,
                                                 ddy,
                                                 dip_subfallas_radianes[j_ny],
                                                 1000*prof_usada[j_ny],
                                                 width_subfallas,
                                                 length_subfallas,
                                                 rake_usado[j_ny],
                                                 np.radians(strike_subfallas_deg[j_ny])
                                                 )
                    except ValueError:
                    ##width variable                                    

                        ve,vn,vz = desplaz_okada(
                                                 ddx,
                                                 ddy,
                                                 dip_subfallas_radianes[j_ny],
                                                 1000*prof_usada[j_ny],
                                                 width_subfallas[j_ny],
                                                 length_subfallas,
                                                 rake_usado[j_ny],
                                                 np.radians(strike_subfallas_deg[j_ny])
                                                 )
                                            
                    if ndim == 1:                
                       M = ve
                    elif ndim == 2:
                       M = np.hstack((ve,vn))
                    elif ndim == 3:
                       M = np.hstack((ve,vn,vz))

                else:
                   
                    ddx,ddy = self.proj_mesh2okada(lat_vertices_subfallas_slab[subfalla][0],
                                                   lon_vertices_subfallas_slab[subfalla][0],
                                                   np.radians(strike_subfallas_deg[j_ny]),
                                                   melnick=False)
                   
                    try:
                    ## dip constante
                        ve,vn,vz = desplaz_okada(ddx,
                                             ddy,
                                             dip_subfallas_radianes[j_ny],
                                             1000*prof_usada[j_ny],
                                             width_subfallas,
                                             length_subfallas,
                                             rake_usado[j_ny],
                                             np.radians(strike_subfallas_deg[j_ny])
                                             )
                    except ValueError:
                    ## dip variable

                        ve,vn,vz = desplaz_okada(
                                                 ddx,
                                                 ddy,
                                                 dip_subfallas_radianes[j_ny],
                                                 1000*prof_usada[j_ny],
                                                 width_subfallas[j_ny],
                                                 length_subfallas,
                                                 rake_usado[j_ny],
                                                 np.radians(strike_subfallas_deg[j_ny])
                                                 )
                                            
                    if self.n_dim == 1 :                
                        tempstack= ve
                    elif self.n_dim == 2 :
                        tempstack = np.hstack((ve,vn))
                    elif self.n_dim == 3 :
                        tempstack=  np.hstack((ve,vn,vz))
                    M = np.vstack((M,tempstack))
                   
                subfalla=subfalla+1
        
        A= M.T
       
        if self.data_insar:
            subfalla=0

            for j_ny in range(ny):
                for i_nx in range(nx):
                    if i_nx == 0  and j_ny == 0:
                        ddx,ddy= self.proj_mesh2okada(lat_vertices_subfallas_slab[subfalla][0],
                                                      lon_vertices_subfallas_slab[subfalla][0],
                                                      np.radians(strike_subfallas_deg[j_ny]),
                                                      insar=True)

                        ulos = self.desplaz_okada_insar(ddx,
                                         ddy,
                                         dip_subfallas_radianes[j_ny],
                                         1000*prof_usada[j_ny],
                                         width_subfallas[j_ny],
                                         length_subfallas,
                                         rake_usado[j_ny],
                                         np.radians(strike_subfallas_deg[j_ny])
                                         )
                        M=ulos
                    else:
                        ddx,ddy= self.proj_mesh2okada(lat_vertices_subfallas_slab[subfalla][0],
                                              lon_vertices_subfallas_slab[subfalla][0],
                                              np.radians(strike_subfallas_deg[j_ny]),
                                              insar=True)

                        ulos = self.desplaz_okada_insar(ddx,
                                         ddy,
                                         dip_subfallas_radianes[j_ny],
                                         1000*prof_usada[j_ny],
                                         width_subfallas[j_ny],
                                         length_subfallas,
                                         rake_usado[j_ny],
                                         np.radians(strike_subfallas_deg[j_ny]))
                        M=np.vstack((M,ulos))
                    subfalla+=1

        if save_GF:
            

            stations=[elem.get('Station') for elem in self.data_estaciones]
            A_L=A.tolist()
            first_lon_vert=["{0:.4f}".format(x[0]) for x in lon_vertices_subfallas_slab ]
            first_lat_vert=["{0:.4f}".format(x[0]) for x in lat_vertices_subfallas_slab ]

            if self.n_dim==3:
                nsta=int(len(A_L)/3)
                if nx*ny > 1:
                    for i in range(nsta):
                        if nx*ny > 1:    
                            GE=["{0:.8f}".format(x) for x in A_L[i]]
                            GN=["{0:.8f}".format(x) for x in A_L[i+nsta]]
                            GZ=["{0:.8f}".format(x) for x in A_L[i+2*nsta]]
                        
                    
                        gf_output='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf/'

                        with open(gf_output+'GF_'+falla[:3].upper()+'_'+label+'_'+stations[i]+'.txt','w') as csvfile:
                            writer=csv.writer(csvfile,delimiter=';')
                            writer.writerow(['lon_subfalla','lat_subfalla','GE','GN','GZ'])
                            for row in zip(first_lon_vert,
                                           first_lat_vert,
                                           GE,
                                           GN,
                                           GZ):
                                writer.writerow(row)
               
                if self.data_insar:
                    M_L=M.T.tolist()
                    gf_output='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf_insar/'
                    np.savetxt(gf_output+'GFLOS_'+falla[:3].upper()+'_'+label+'.txt',M.T)
                

                    self.planos_falla_obj.update({label+'_LOS':M.T})



            

            if self.n_dim==2:
                nsta=int(len(A_L)/2)
                if nx*ny > 1:
                    for i in range(nsta):
                        if nx*ny > 1:    
                            GE=["{0:.8f}".format(x) for x in A_L[i]]
                            GN=["{0:.8f}".format(x) for x in A_L[i+nsta]]
                        
                    
                        gf_output='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf/'

                        with open(gf_output+'GF_'+falla[:3].upper()+'_'+label+'_'+stations[i]+'.txt','w') as csvfile:
                            writer=csv.writer(csvfile,delimiter=';')
                            writer.writerow(['lon_subfalla','lat_subfalla','GE','GN'])
                            for row in zip(first_lon_vert,
                                           first_lat_vert,
                                           GE,
                                           GN):  
                                writer.writerow(row) 
            ## special case needed when 1 subfault for not getting an error
            elif nx*ny == 1:
                        GE=["{0:.8f}".format(x) for x in A_L[:nsta]]
                        GN=["{0:.8f}".format(x) for x in A_L[nsta:2*nsta]]
                        GZ=["{0:.8f}".format(x) for x in A_L[2*nsta:3*nsta]]
                        gf_output='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf/'
                        for i in range(nsta):
                            with open(gf_output+'GF_'+falla[:3].upper()+'_'+label+'_'+stations[i]+'.txt','w') as csvfile:
                                writer=csv.writer(csvfile,delimiter=';')
                                writer.writerow(['lon_subfalla','lat_subfalla','GE','GN','GZ'])
                                writer.writerow([float(first_lon_vert[0]),float(first_lat_vert[0]),GE[i],GN[i],GZ[i]])
    
        
        self.planos_falla_obj.update({label:A})
        return A


    
        
    def load_greenfunctions(self):
        #path_gf='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf/'.replace(self.output,'direct')
        path_gf='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf/'
        path_gf_insar='Outputs'+ self.id_proceso+'/'+self.proceso+'/'+self.interface+'/'+self.output+'/gf_insar/'

        if self.output == 'inversion' and self.synthetic_test:
            path_gf=path_gf.replace('inversion','direct')
            path_gf_insar=path_gf_insar.replace('inversion','direct')
       
        gfs=sorted(os.listdir(path_gf))
        nsta=[x for x in gfs if '_INV_A' in x]
        nx=self.grid_size[0]
        ny=self.grid_size[1]

        if self.proceso == 'interseismic':
            
            ## LOAD SAVAGE INTERSEISMIC GF
            if self.interface == 'single':
                gfs = [path_gf+x for  x in gfs if '_B_' not in x and '_C_' not in x and '_D_' not in x and '_E_' not in x  ]
                Aeste=[]
                Anorte=[]
                Aup=[]
                lon_falla_ab=[]
                lat_falla_ab=[]
                bandera=True
                for gf in gfs:
                    with open(gf,'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter =';')
                        if 'gf/GF_NOR_A' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Aeste.append(row[2])
                                    Anorte.append(row[3])
                                    if bandera:
                                        lon_falla_ab.append(row[0])
                                        lat_falla_ab.append(row[1])
                    bandera = False
                Aeste=np.asarray(Aeste).reshape((len(gfs),nx*ny))
                Anorte=np.asarray(Anorte).reshape((len(gfs),nx*ny))
                A=np.vstack((Aeste,Anorte))
                G={'A':A}
                coord_fallas={'AB':(lon_falla_ab,lat_falla_ab)}

                   

            ## LOAD ABCDE GFS
            elif self.interface == 'double':
                ## we just load all green functions!
                gfs = [path_gf+x for  x in gfs ]

                Aeste=[]
                Anorte=[]
                Aup=[]
                Beste=[]
                Bnorte=[]
                Bup=[]
                Ceste=[]
                Cnorte=[]
                Cup=[]
                Deste=[]
                Dnorte=[]
                Dup=[]
                Eeste=[]
                Enorte=[]
                Eup=[]

                lon_falla_ab=[]
                lat_falla_ab=[]
                lon_falla_cd=[]
                lat_falla_cd=[]
                lon_falla_e=[]
                lat_falla_e=[]

                datos=len(gfs)
                nx=self.grid_size[0]
                ny=self.grid_size[1]
                bandera = True
                for gf in gfs:
                    with open(gf,'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter =';')
                        if 'gf/GF_INV_A' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Aeste.append(row[2])
                                    Anorte.append(row[3])
                                    if bandera:
                                        lon_falla_ab.append(row[0])
                                        lat_falla_ab.append(row[1])
                                        #bandera = False
                        
                        elif 'gf/GF_NOR_B' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Beste.append(row[2])
                                    Bnorte.append(row[3])

                        elif 'gf/GF_INV_C' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Ceste.append(row[2])
                                    Cnorte.append(row[3])
                                    if bandera:
                                        lon_falla_cd.append(row[0])
                                        lat_falla_cd.append(row[1])

                        elif 'gf/GF_NOR_D' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Deste.append(row[2])
                                    Dnorte.append(row[3])

                        elif 'gf/GF_NOR_E' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Eeste.append(row[2])
                                    Enorte.append(row[3])
                                    if bandera:
                                        lon_falla_e.append(row[0])
                                        lat_falla_e.append(row[1])
                    bandera = False

                Aeste=np.asarray(Aeste).reshape((int(len(nsta)),int(nx*ny)))
                Anorte=np.asarray(Anorte).reshape((int(len(nsta)),int(nx*ny)))

                Beste=np.asarray(Beste).reshape((int(len(nsta)),int(nx*ny)))
                Bnorte=np.asarray(Bnorte).reshape((int(len(nsta)),int(nx*ny)))

                Ceste=np.asarray(Ceste).reshape((int(len(nsta)),int(nx*ny)))
                Cnorte=np.asarray(Cnorte).reshape((int(len(nsta)),int(nx*ny)))
                
                Deste=np.asarray(Deste).reshape((int(len(nsta)),int(nx*ny)))
                Dnorte=np.asarray(Dnorte).reshape((int(len(nsta)),int(nx*ny)))

                Eeste=np.asarray(Eeste).reshape((int(len(nsta)),nx))
                Enorte=np.asarray(Enorte).reshape((int(len(nsta)),nx))

                A=np.vstack((Aeste,Anorte))
                B=np.vstack((Beste,Bnorte))
                C=np.vstack((Ceste,Cnorte))
                D=np.vstack((Deste,Dnorte))
                E=np.vstack((Eeste,Enorte))
                G={'A':A,
                   'B':B,
                   'C':C,
                   'D':D,
                   'E':E}
                coord_fallas={'AB':(lon_falla_ab,lat_falla_ab),
                              'CD':(lon_falla_cd,lat_falla_cd),
                               'E':(lon_falla_e,lat_falla_e)}



        if self.proceso == 'coseismic':
            if self.interface == 'single':
                gfs = [path_gf+x for  x in gfs if '_B_' not in x and '_C_' not in x and '_D_' not in x and '_E_' not in x  ]
            elif self.interface == 'double':
                gfs = [path_gf+x for  x in gfs if '_C_' not in x and '_D_' not in x and '_E_' not in x  ]

        
            lon_falla_ab=[]
            lat_falla_ab=[]
           
            bandera=True
            if self.interface == 'single' and not self.split_G:
                Aeste=[]
                Anorte=[]
                Aup=[]
                if self.data_melnick:
                     gfs_meln=gfs[-33:]
                     gfs=gfs[:-33]
                for gf in gfs:
                    with open(gf,'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter =';')
                        for row in reader:
                            if row[0][0] != 'l':
                                row =[float(x) for x in row]
                                Aeste.append(row[2])
                                Anorte.append(row[3])
                                Aup.append(row[4])
                                if bandera:
                                    lon_falla_ab.append(row[0])
                                    lat_falla_ab.append(row[1])
                                    ## OJO CON ESTA BANDERITA
                    bandera = False
                

                if nx*ny == 1:
                    Aeste=np.asarray(Aeste).reshape((len(Aeste),nx*ny))
                    Anorte=np.asarray(Anorte).reshape((len(Anorte),nx*ny))
                    Aup=np.asarray(Aup).reshape((len(Aup),nx*ny))
                if nx*ny > 1:
                    Aeste=np.asarray(Aeste).reshape((len(gfs),nx*ny))
                    Anorte=np.asarray(Anorte).reshape((len(gfs),nx*ny))
                    Aup=np.asarray(Aup).reshape((len(gfs),nx*ny))
                A=np.vstack((Aeste,Anorte,Aup))
                G={'A':A}
                coord_fallas={'AB':(lon_falla_ab,lat_falla_ab)}

               
                if self.data_insar:
                    Alos=np.loadtxt(path_gf_insar+'GFLOS_INV_A.txt')
                    G.update({'ALOS':Alos})
               

            if self.interface == 'double':
                Aeste=[]
                Anorte=[]
                Aup=[]
                Beste=[]
                Bnorte=[]
                Bup=[]
                nx=self.grid_size[0]
                ny=self.grid_size[1]
                for gf in gfs:
                    with open(gf,'r') as csv_file:
                        reader = csv.reader(csv_file, delimiter =';')
                        if 'gf/GF_INV_A' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Aeste.append(row[2])
                                    Anorte.append(row[3])
                                    Aup.append(row[4])
                                    if bandera:
                                        lon_falla_ab.append(row[0])
                                        lat_falla_ab.append(row[1])
                                        #bandera = False
                        
                        elif 'gf/GF_NOR_B' in gf:
                            for row in reader:
                                if row[0][0] != 'l':
                                    row = [float(x) for x in row]
                                    Beste.append(row[2])
                                    Bnorte.append(row[3])
                                    Bup.append(row[4])
                                    # if bandera:
                                    #     lon_falla.append(row[0])
                                    #     lat_falla.append(row[1])
                    bandera = False
                if nx*ny == 1:
                    Aeste=np.asarray(Aeste).reshape((len(Aeste),nx*ny))
                    Anorte=np.asarray(Anorte).reshape((len(Anorte),nx*ny))
                    Aup=np.asarray(Aup).reshape((len(Aup),nx*ny))

                    Beste=np.asarray(Beste).reshape((len(Beste),nx*ny))
                    Bnorte=np.asarray(Bnorte).reshape((len(Bnorte),nx*ny))
                    Bup=np.asarray(Bup).reshape((len(Bup),nx*ny))

                if nx*ny > 1:
                    Aeste=np.asarray(Aeste).reshape((int(len(gfs)/2),nx*ny))
                    Anorte=np.asarray(Anorte).reshape((int(len(gfs)/2),nx*ny))
                    Aup=np.asarray(Aup).reshape((int(len(gfs)/2),nx*ny))

                    Beste=np.asarray(Beste).reshape((int(len(gfs)/2),nx*ny))
                    Bnorte=np.asarray(Bnorte).reshape((int(len(gfs)/2),nx*ny))
                    Bup=np.asarray(Bup).reshape((int(len(gfs)/2),nx*ny))



                A=np.vstack((Aeste,Anorte,Aup))
                B=np.vstack((Beste,Bnorte,Bup))
                G=np.hstack((A,B))
                #G=np.hstack((Aeste,Anorte,Aup,Beste,Bnorte,Bup)).reshape(int(len(Aeste)*3),2)
                #coord_fallas=(lon_falla,lat_falla)
                G={'A':A,'B':B}
                coord_fallas={'AB':(lon_falla_ab,lat_falla_ab)}
          
        if self.interface == 'single' and self.split_G:
            Ge_ss=[]
            Gn_ss=[]
            Gz_ss=[]
            Ge_ds=[]
            Gn_ds=[]
            Gz_ds=[]
            datos=len(gfs)
            nx=self.grid_size[0]
            ny=self.grid_size[1]
            for gf in gfs:
                with open(gf,'r') as csv_file:
                    reader = csv.reader(csv_file, delimiter =';')
                    if 'gf/GF_INV' in gf:
                        for row in reader:
                            if row[0][0] != 'l':
                                row = [float(x) for x in row]
                                Ge_ss.append(row[2])
                                Gn_ss.append(row[3])
                                Gz_ss.append(row[4])
                                Ge_ds.append(row[5])
                                Gn_ds.append(row[6])
                                Gz_ds.append(row[7])

                                if bandera:
                                    lon_falla.append(row[0])
                                    lat_falla.append(row[1])
                                        #bandera = False
                        
                       
                bandera = False
        
            Ge_ss=np.asarray(Ge_ss).reshape((int(datos),nx*ny))
            Gn_ss=np.asarray(Gn_ss).reshape((int(datos),nx*ny))
            Gz_ss=np.asarray(Gz_ss).reshape((int(datos),nx*ny))

            Ge_ds=np.asarray(Ge_ds).reshape((int(datos),nx*ny))
            Gn_ds=np.asarray(Gn_ds).reshape((int(datos),nx*ny))
            Gz_ds=np.asarray(Gz_ds).reshape((int(datos),nx*ny))



            Gss=np.vstack((Ge_ss,Gn_ss,Gz_ss))
            Gds=np.vstack((Ge_ds,Gn_ds,Gz_ds))
            G=np.hstack((Gss,Gds))

            coord_fallas=(lon_falla,lat_falla)
            

        return G,coord_fallas
        

    def make_directo_new(self,model='slabmodel',**kwargs):
        nx=self.grid_size[0]
        ny=self.grid_size[1]

        slip_paper=kwargs.get('load_paper',True)
       
        ## COSEISMIC CASE
        if self.proceso == 'coseismic':
            A = self.model_matrix_slab(falla='inversa',save_subfallas_methods=True,
                                  generar_subfallas = True,
                                  save_GF=True)

            if self.data_insar:
                Alos=self.planos_falla_obj.get('A_LOS')
                A=np.vstack((A,Alos))
            slip_A = kwargs.get('slipA',np.ones(self.grid_size[0]*self.grid_size[1]))
            if slip_paper:
                self.interpolate_slip()
                slip_A=self.slip_sintetico.get('cosismico_thrust')
                slip_A=slip_A


            ## COSEISMIC CASE: A B PLANE CASE
            if self.interface == 'double':

                B = self.model_matrix_slab(falla='normal',save_subfallas_methods=True,
                                  generar_subfallas = True,
                                  save_GF=True)

                if self.data_insar:
                    Blos=self.planos_falla_obj.get('B_LOS')
                    B=np.vstack((B,Blos))
                

                slip_B = kwargs.get('slipB',np.ones(nx*ny))
                if slip_paper:
                    self.interpolate_slip()
                    slip_B=self.slip_sintetico.get('cosismico_normal')
                slip=np.hstack((slip_A,slip_B)).reshape(2*nx*ny,1)

                if nx*ny==1:
                    G=np.vstack((A,B)).T
                else:
                    G=np.hstack((A,B))

                ## anteponemos un mil para que el resultado este en milimetros
                ## Hacemos U = Gs
                daux=(1000*np.matmul(G,slip))
                dobs = [x[0] for x in daux]

            ## define the input slip that will be applied to the system:

            ## COSEISMIC CASE: A PLANE 
            if self.interface == 'single':
                try :
                    slip = slip_A
                    G=A
                    dobs = (1000*np.matmul(G,slip)).tolist()

                except ValueError:
                    print('solo una subfalla, arreglando error...')
                    slip = slip_A
                    G=A.reshape(A.shape[0],1)
                    ## anteponemos un mil para que el resultado este en milimetros
                    ## Hacemos U = Gs
                    dobs=(1000*np.matmul(G,slip)).tolist()

        ## GENERAL PROCESSING FOR ALL CASES
        Ue_teo=[]
        Un_teo=[]
        Uz_teo=[]
        if not self.data_insar:
            nsta=int(len(dobs)/self.n_dim)
        if self.data_insar:
            #nsta=int((len(dobs)-(Alos.shape)[0])/3))
            nsta=int((len(dobs)-Alos.shape[0])/3)
        if self.proceso=='coseismic':
            for i in range(nsta):
                Ue_teo.append(dobs[i])
                Un_teo.append(dobs[i+nsta])
                Uz_teo.append(dobs[i+2*nsta])


            self.data_sintetica.update({'UE': Ue_teo,
                                        'UN': Un_teo,
                                        'UU': Uz_teo})

            if self.data_insar:
                Ulos=dobs[-Alos.shape[0]:]
                self.data_sintetica.update({'ULOS':Ulos})

                lon_i=[elem.get('Longitud') for elem in self.data_estaciones_insar]
                lat_i = [elem.get('Latitud') for elem in self.data_estaciones_insar]
                slo_i = [1000*elem.get('SLO') for elem in self.data_estaciones_insar]
                re=[elem.get('EE') for elem in self.data_estaciones_insar]
                rn=[elem.get('NN') for elem in self.data_estaciones_insar]
                ru=[elem.get('UU') for elem in self.data_estaciones_insar]
                TOPO=[elem.get('TOPO') for elem in self.data_estaciones_insar]

                
            lon = [elem.get('Longitud') for elem in self.data_estaciones]
            lat = [elem.get('Latitud') for elem in self.data_estaciones]
            sta = [elem.get('Station') for elem in self.data_estaciones]
            SE = [1000*elem.get('SE') for elem in self.data_estaciones]
            SN = [1000*elem.get('SN') for elem in self.data_estaciones]
            SU = [1000*elem.get('SU') for elem in self.data_estaciones]      
            
        if self.proceso=='interseismic':
            for i in range(nsta):
                Ue_teo.append(dobs[i])
                Un_teo.append(dobs[i+nsta])

            self.data_sintetica.update({'UE': Ue_teo,
                                        'UN': Un_teo})
                        

            lon = [elem.get('Longitud') for elem in self.data_estaciones]
            lat = [elem.get('Latitud') for elem in self.data_estaciones]
            sta = [elem.get('Station') for elem in self.data_estaciones]
            SE = [1000*elem.get('SE') for elem in self.data_estaciones]
            SN = [1000*elem.get('SN') for elem in self.data_estaciones]
       


        ## archivo de movimientos sinteticos
            
        name_archivo = self.desplaz_output
        name_slips_upper = self.upper_interface_output
        name_slips_lower = self.lower_interface_output
            
        lon_central = self.subfallas_model_dict.get('lon_central')
        lat_central = self.subfallas_model_dict.get('lat_central')
            

        ## central latitude of the subfaults  of the fault plane (will work for both)
        ## DATA IS ON MILIMETERS

        if self.proceso=='coseismic':
            with open(name_archivo,'w') as csvfile:
                writer=csv.writer(csvfile,delimiter=';')
                writer.writerow( ['Station','Longitud','Latitud'] + list(self.data_sintetica.keys()) +['SE','SN','SU'])
                for row in zip(sta,lon,lat,Ue_teo,Un_teo,Uz_teo,SE,SN,SU):
                    writer.writerow(row)

            if self.data_insar:
                name_archivo_insar = self.desplaz_output_insar
                with open(name_archivo_insar,'w') as csvfile:
                    writer=csv.writer(csvfile,delimiter=';')
                    writer.writerow( ['Longitud','Latitud','LOS','SLO','EE','NN','UU','TOPO'])
                    for row in zip(lon_i,lat_i,Ulos,slo_i,re,rn,ru,TOPO):
                        writer.writerow(row)

        if self.proceso=='interseismic':
            with open(name_archivo,'w') as csvfile:
                writer=csv.writer(csvfile,delimiter=';')
                writer.writerow( ['Station','Longitud','Latitud','UE','UN']  +['SE','SN'])
                for row in zip(sta,lon,lat,Ue_teo,Un_teo,SE,SN):
                    writer.writerow(row)

        shutil.copy(name_archivo,'./Fuentes/')

        if self.proceso == 'interseismic':
            slip_key='slip_rate'
        if self.proceso == 'coseismic':
            slip_key='slip'

        with open(name_slips_upper,'w') as csvfile:
            writer=csv.writer(csvfile,delimiter=';')
            writer.writerow( ['lon_central','lat_central',slip_key+'_upper'] )
            for row in zip(lon_central,lat_central,slip_A):
                writer.writerow(row)
                    
        if self.interface == 'double':
            with open(name_slips_lower,'w') as csvfile:
                writer=csv.writer(csvfile,delimiter=';')
                writer.writerow( ['lon_central','lat_central',slip_key+'_lower'] )
                for row in zip(lon_central,lat_central,slip_B):
                    writer.writerow(row)
        dx=self.subfallas_model_dict.get('length_model')
        width=np.mean(self.subfallas_model_dict.get('width_model'))
        return dobs,G,dx,width



   

    def gen_matriz_suavidad_2(self):
        if self.interface=='single':
            nx=self.grid_size[0]
            ny=self.grid_size[1]
            x2=np.arange(1,20)**2
            res=x2-nx*ny
            if all(res):
                L=laplacian2D(int(np.sqrt(nx*ny)+1)).toarray()
                L=L[:nx*ny,:nx*ny]
            if not all(res):
                L=laplacian2D(int(np.sqrt(nx*ny))).toarray()

        if self.interface=='double':
            nx=self.grid_size[0]
            ny=self.grid_size[1]
            x2=np.arange(1,20)**2
            res=x2-nx*ny
            if all(res):
                L=laplacian2D(int(np.sqrt(nx*ny)+1)).toarray()
                L=L[:nx*ny,:nx*ny]
            if not all(res):
                L=laplacian2D(int(np.sqrt(nx*ny))).toarray()
            
        return L


    def desplaz_okada_insar(self,xi,yi,dip,d,w,l,rake,strike):
    # notacion  de Chinnery: f(e,eta)||= f(x,p)-f(x,p-W)-f(x-L,p)+f(x-L,W-p)
    # this functions generates green functions in every fault
        lembda,mu=lame(E=100,po_ratio=0.25)

        re=1*np.array([elem.get('EE') for elem in self.data_estaciones_insar])
        rn=np.array([elem.get('NN') for elem in self.data_estaciones_insar])
        ru=np.array([elem.get('UU') for elem in self.data_estaciones_insar])
        los=np.array([elem.get('LOS') for elem in self.data_estaciones_insar])
        sign=np.array([-1.0 if x < 0.0 else 1.0 for x in los ])


        p = yi*np.cos(dip) + d*np.sin(dip)                                       
        q = yi*np.sin(dip) - d*np.cos(dip)
        e = np.array([xi,xi,xi-l,xi-l]).T
        eta = np.array([p,p-w,p,p-w]).T       
        #qq array de q, hay cuatro valores porque se opera con los cuatro valores de eta.               
        qq = np.array([q,q,q,q]).T                    
        ytg = eta*np.cos(dip) + qq*np.sin(dip)           
        dtg = eta*np.sin(dip) - qq*np.cos(dip)           
        R = np.power(e**2 + eta**2 + qq**2, 0.5)     
        X = np.power(e**2 + qq**2, 0.5)               

        I5 = (2*mu/(lembda+mu)/np.cos(dip))*scp.arctan((eta*(X+qq*np.cos(dip))+X*(R+X)*np.sin(dip))/(e*(R+X)*np.cos(dip))) 
        I4 = mu/(lembda+mu)/np.cos(dip)*(scp.log(R+dtg)-np.sin(dip)*scp.log(R+eta)) 
        # Solido de Poisson mu=lambda
        I1 = mu/(lembda+mu)*((-1./np.cos(dip))*(e/(R+dtg)))-(np.sin(dip)*I5/np.cos(dip)) 
        I3 = mu/(lembda+mu)*(1/np.cos(dip)*(ytg/(R+(dtg)))-scp.log(R+eta))+(np.sin(dip)*I4/np.cos(dip))  
        I2 = mu/(lembda+mu)*(-scp.log(R+eta))-I3   #ok

        #dip-slip  (en direccion del manteo)
        ux_ds = -np.sin(rake)/(2*np.pi)*(qq/R-I3*np.sin(dip)*np.cos(dip))  #o
        uy_ds = -np.sin(rake)/(2*np.pi)*((ytg*qq/R/(R+e))+(np.cos(dip)*scp.arctan(e*eta/qq/R))
                    -(I1*np.sin(dip)*np.cos(dip))) 
        uz_ds = -np.sin(rake)/(2*np.pi)*((dtg*qq/R/(R+e))+(np.sin(dip)*scp.arctan(e*eta/qq/R))
                    -(I5*np.sin(dip)*np.cos(dip)))

        

        # strike-slip  (en direccion del strike)
        ux_ss = -np.cos(rake)/(2*np.pi)*((e*qq/R/(R+eta))+(scp.arctan(e*eta/(qq*R)))+I1*np.sin(dip))  
        uy_ss = -np.cos(rake)/(2*np.pi)*((ytg*qq/R/(R+eta))+qq*np.cos(dip)/(R+eta)+I2*np.sin(dip))       
        uz_ss = -np.cos(rake)/(2*np.pi)*((dtg*qq/R/(R+eta))+qq*np.sin(dip)/(R+eta)+I4*np.sin(dip))
            
        # representacion chinnery dip-slip
        Gxd = ux_ds.T[0]-ux_ds.T[1]-ux_ds.T[2]+ux_ds.T[3]   
        Gyd = uy_ds.T[0]-uy_ds.T[1]-uy_ds.T[2]+uy_ds.T[3]   
        Gzd = uz_ds.T[0]-uz_ds.T[1]-uz_ds.T[2]+uz_ds.T[3]   

        # representacion chinnery strike-slip
        Gxs = ux_ss.T[0]-ux_ss.T[1]-ux_ss.T[2]+ux_ss.T[3]  
        Gys = uy_ss.T[0]-uy_ss.T[1]-uy_ss.T[2]+uy_ss.T[3]  
        Gzs = uz_ss.T[0]-uz_ss.T[1]-uz_ss.T[2]+uz_ss.T[3]
        ## sumar strike slip y dip slip
        Gx=(Gxs+Gxd)
        Gy=(Gys+Gyd)
        Gz=(Gzs+Gzd)
        # proyeccion a las componentes geograficas
        # rotacion de sistema de coordenadas 
        # y multiplicacion por vector unitario LOS 
        Ge = re*(Gx*np.sin(strike) - Gy*np.cos(strike)) 
        Gn = rn*(Gx*np.cos(strike) + Gy*np.sin(strike)) 
        Gz = ru*Gz
        # funcion de green insar
        G=Ge+Gn+Gz
        #G=sign*np.sqrt(Ge**2+Gn**2+Gz**2)
        #print('ka',Ge,Gn,Gz)
        #print('total',G)

        return G






def desplaz_okada(xi,yi,dip,d,w,l,rake,strike):
    # notacion  de Chinnery: f(e,eta)||= f(x,p)-f(x,p-W)-f(x-L,p)+f(x-L,W-p)
    # this functions generates green functions in every fault
    lembda,mu=lame(E=100,po_ratio=0.25)
    p = yi*np.cos(dip) + d*np.sin(dip)                                       
    q = yi*np.sin(dip) - d*np.cos(dip)
    e = np.array([xi,xi,xi-l,xi-l]).T
    eta = np.array([p,p-w,p,p-w]).T       
    #qq array de q, hay cuatro valores porque se opera con los cuatro valores de eta.               
    qq = np.array([q,q,q,q]).T                    
    ytg = eta*np.cos(dip) + qq*np.sin(dip)           
    dtg = eta*np.sin(dip) - qq*np.cos(dip)           
    R = np.power(e**2 + eta**2 + qq**2, 0.5)     
    X = np.power(e**2 + qq**2, 0.5)               

    I5 = (2*mu/(lembda+mu)/np.cos(dip))*scp.arctan((eta*(X+qq*np.cos(dip))+X*(R+X)*np.sin(dip))/(e*(R+X)*np.cos(dip))) 
    I4 = mu/(lembda+mu)/np.cos(dip)*(scp.log(R+dtg)-np.sin(dip)*scp.log(R+eta)) 
    # Solido de Poisson mu=lambda
    I1 = mu/(lembda+mu)*((-1./np.cos(dip))*(e/(R+dtg)))-(np.sin(dip)*I5/np.cos(dip)) 
    I3 = mu/(lembda+mu)*(1/np.cos(dip)*(ytg/(R+(dtg)))-scp.log(R+eta))+(np.sin(dip)*I4/np.cos(dip))  
    I2 = mu/(lembda+mu)*(-scp.log(R+eta))-I3   #ok

    #dip-slip  (en direccion del manteo)
    ux_ds = -np.sin(rake)/(2*np.pi)*(qq/R-I3*np.sin(dip)*np.cos(dip))  #o
    uy_ds = -np.sin(rake)/(2*np.pi)*((ytg*qq/R/(R+e))+(np.cos(dip)*scp.arctan(e*eta/qq/R))
                -(I1*np.sin(dip)*np.cos(dip))) 
    uz_ds = -np.sin(rake)/(2*np.pi)*((dtg*qq/R/(R+e))+(np.sin(dip)*scp.arctan(e*eta/qq/R))
                -(I5*np.sin(dip)*np.cos(dip)))

   
    ux_ss = -np.cos(rake)/(2*np.pi)*((e*qq/R/(R+eta))+(scp.arctan(e*eta/(qq*R)))+I1*np.sin(dip))  
    uy_ss = -np.cos(rake)/(2*np.pi)*((ytg*qq/R/(R+eta))+qq*np.cos(dip)/(R+eta)+I2*np.sin(dip))       
    uz_ss = -np.cos(rake)/(2*np.pi)*((dtg*qq/R/(R+eta))+qq*np.sin(dip)/(R+eta)+I4*np.sin(dip))
        
    # representacion chinnery dip-slip
    uxd = ux_ds.T[0]-ux_ds.T[1]-ux_ds.T[2]+ux_ds.T[3]   
    uyd = uy_ds.T[0]-uy_ds.T[1]-uy_ds.T[2]+uy_ds.T[3]   
    uzd = uz_ds.T[0]-uz_ds.T[1]-uz_ds.T[2]+uz_ds.T[3]   

    # representacion chinnery strike-slip
    uxs = ux_ss.T[0]-ux_ss.T[1]-ux_ss.T[2]+ux_ss.T[3]  
    uys = uy_ss.T[0]-uy_ss.T[1]-uy_ss.T[2]+uy_ss.T[3]  
    uzs = uz_ss.T[0]-uz_ss.T[1]-uz_ss.T[2]+uz_ss.T[3]  

    # solucion 
    ux = uxd+uxs  
    uy = uyd+uys  
    uz = uzd+uzs
   

    Ue = ux*np.sin(strike) - uy*np.cos(strike) 
    Un = ux*np.cos(strike) + uy*np.sin(strike) 
    #El gran problema es que falta el vector de deslizamiento, pero se podria agregar despues (multiplicando)

    return Ue,Un,uz


def get_rake( Rumbo_placa , strike_subfalla,dip_subfalla):
        s =  strike_subfalla 
        d =  dip_subfalla 
        r =  Rumbo_placa 
        
        seno_rake = ( np.sin(s) * np.cos(r) - np.cos(s) * np.sin(r) ) / np.cos(d)
        cose_rake = np.cos(s) * np.cos(r) + np.sin(s) * np.sin(r)
        
        rake = np.arctan2( seno_rake, cose_rake ) + np.pi 
        
        return  rake 

def vinc_dist(  phi1,  lembda1,  phi2,  lembda2 ) :
        """ 
        Returns the distance between two geographic points on the ellipsoid
        and the forward and reverse azimuths between these points.
        lats, longs and azimuths are in decimal degrees, distance in metres 

        Returns ( s, alpha12,  alpha21 ) as a tuple
        """

        f = 1.0 / 298.257223563     # WGS84
        a = 6378137.0           # metres
        if (abs( phi2 - phi1 ) < 1e-8) and ( abs( lembda2 - lembda1) < 1e-8 ) :
                return 0.0, 0.0, 0.0

        piD4   = math.atan( 1.0 )
        two_pi = piD4 * 8.0

        phi1    = phi1 * piD4 / 45.0
        lembda1 = lembda1 * piD4 / 45.0     # unfortunately lambda is a key word!
        phi2    = phi2 * piD4 / 45.0
        lembda2 = lembda2 * piD4 / 45.0

        b = a * (1.0 - f)

        TanU1 = (1-f) * math.tan( phi1 )
        TanU2 = (1-f) * math.tan( phi2 )

        U1 = math.atan(TanU1)
        U2 = math.atan(TanU2)

        lembda = lembda2 - lembda1
        last_lembda = -4000000.0        # an impossibe value
        omega = lembda

        # Iterate the following equations, 
        #  until there is no significant change in lembda 

        while ( last_lembda < -3000000.0 or lembda != 0 and abs( (last_lembda - lembda)/lembda) > 1.0e-9 ) :

                sqr_sin_sigma = pow( np.cos(U2) * np.sin(lembda), 2) + \
                        pow( (np.cos(U1) * np.sin(U2) - \
                        np.sin(U1) *  np.cos(U2) * np.cos(lembda) ), 2 )

                Sin_sigma = np.sqrt( sqr_sin_sigma )

                Cos_sigma = np.sin(U1) * np.sin(U2) + np.cos(U1) * np.cos(U2) * np.cos(lembda)
        
                sigma = math.atan2( Sin_sigma, Cos_sigma )

                Sin_alpha = np.cos(U1) * np.cos(U2) * np.sin(lembda) / np.sin(sigma)
                alpha = math.asin( Sin_alpha )

                Cos2sigma_m = np.cos(sigma) - (2 * np.sin(U1) * np.sin(U2) / pow(np.cos(alpha), 2) )

                C = (f/16) * pow(np.cos(alpha), 2) * (4 + f * (4 - 3 * pow(np.cos(alpha), 2)))

                last_lembda = lembda

                lembda = omega + (1-C) * f * np.sin(alpha) * (sigma + C * np.sin(sigma) * \
                        (Cos2sigma_m + C * np.cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2) )))

        u2 = pow(np.cos(alpha),2) * (a*a-b*b) / (b*b)

        A = 1 + (u2/16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))

        B = (u2/1024) * (256 + u2 * (-128+ u2 * (74 - 47 * u2)))

        delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B/4) * \
                (Cos_sigma * (-1 + 2 * pow(Cos2sigma_m, 2) ) - \
                (B/6) * Cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * \
                (-3 + 4 * pow(Cos2sigma_m,2 ) )))

        s = b * A * (sigma - delta_sigma)

        alpha12 = math.atan2( (np.cos(U2) * np.sin(lembda)), \
                (np.cos(U1) * np.sin(U2) - np.sin(U1) * np.cos(U2) * np.cos(lembda)))

        alpha21 = math.atan2( (np.cos(U1) * np.sin(lembda)), \
                (-np.sin(U1) * np.cos(U2) + np.cos(U1) * np.sin(U2) * np.cos(lembda)))

        if ( alpha12 < 0.0 ) : 
                alpha12 =  alpha12 + two_pi
        if ( alpha12 > two_pi ) : 
                alpha12 = alpha12 - two_pi

        alpha21 = alpha21 + two_pi / 2.0
        if ( alpha21 < 0.0 ) : 
                alpha21 = alpha21 + two_pi
        if ( alpha21 > two_pi ) : 
                alpha21 = alpha21 - two_pi

        alpha12    = alpha12    * 45.0 / piD4
        alpha21    = alpha21    * 45.0 / piD4
        return s, alpha12,  alpha21 

   # END of Vincenty's Inverse formulae 


#-------------------------------------------------------------------------------
# Vincenty's Direct formulae                            |
# Given: latitude and longitude of a point (phi1, lembda1) and          |
# the geodetic azimuth (alpha12)                        |
# and ellipsoidal distance in metres (s) to a second point,         |
#                                       |
# Calculate: the latitude and longitude of the second point (phi2, lembda2)     |
# and the reverse azimuth (alpha21).                        |
#                                       |
#-------------------------------------------------------------------------------

def  vinc_pt( phi1, lembda1, alpha12, s ) :
        """

        Returns the lat and long of projected point and reverse azimuth
        given a reference point and a distance and azimuth to project.
        lats, longs and azimuths are passed in decimal degrees

        Returns ( phi2,  lambda2,  alpha21 ) as a tuple 

        """
 
        f = 1.0 / 298.257223563     # WGS84
        a = 6378137.0           # metres
        piD4 = math.atan( 1.0 )
        two_pi = piD4 * 8.0

        phi1    = phi1    * piD4 / 45.0
        lembda1 = lembda1 * piD4 / 45.0
        alpha12 = alpha12 * piD4 / 45.0
        if ( alpha12 < 0.0 ) : 
                alpha12 = alpha12 + two_pi
        if ( alpha12 > two_pi ) : 
                alpha12 = alpha12 - two_pi

        b = a * (1.0 - f)

        TanU1 = (1-f) * math.tan(phi1)
        U1 = math.atan( TanU1 )
        sigma1 = math.atan2( TanU1, np.cos(alpha12) )
        Sinalpha = np.cos(U1) * np.sin(alpha12)
        cosalpha_sq = 1.0 - Sinalpha * Sinalpha

        u2 = cosalpha_sq * (a * a - b * b ) / (b * b)
        A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * \
                (320 - 175 * u2) ) )
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2) ) )

        # Starting with the approximation
        sigma = (s / (b * A))

        last_sigma = 2.0 * sigma + 2.0  # something impossible

        # Iterate the following three equations 
        #  until there is no significant change in sigma 

        # two_sigma_m , delta_sigma
        two_sigma_m = 0
        while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ) :
                two_sigma_m = 2 * sigma1 + sigma

                delta_sigma = B * np.sin(sigma) * ( np.cos(two_sigma_m) \
                        + (B/4) * (np.cos(sigma) * \
                        (-1 + 2 * pow( np.cos(two_sigma_m), 2 ) -  \
                        (B/6) * np.cos(two_sigma_m) * \
                        (-3 + 4 * pow(np.sin(sigma), 2 )) *  \
                        (-3 + 4 * pow( np.cos (two_sigma_m), 2 ))))) \

                last_sigma = sigma
                sigma = (s / (b * A)) + delta_sigma

        phi2 = math.atan2 ( (np.sin(U1) * np.cos(sigma) + np.cos(U1) * np.sin(sigma) * np.cos(alpha12) ), \
                ((1-f) * np.sqrt( pow(Sinalpha, 2) +  \
                pow(np.sin(U1) * np.sin(sigma) - np.cos(U1) * np.cos(sigma) * np.cos(alpha12), 2))))

        lembda = math.atan2( (np.sin(sigma) * np.sin(alpha12 )), (np.cos(U1) * np.cos(sigma) -  \
                np.sin(U1) *  np.sin(sigma) * np.cos(alpha12)))

        C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq ))

        omega = lembda - (1-C) * f * Sinalpha *  \
                (sigma + C * np.sin(sigma) * (np.cos(two_sigma_m) + \
                C * np.cos(sigma) * (-1 + 2 * pow(np.cos(two_sigma_m),2) )))

        lembda2 = lembda1 + omega

        alpha21 = math.atan2 ( Sinalpha, (-np.sin(U1) * np.sin(sigma) +  \
                np.cos(U1) * np.cos(sigma) * np.cos(alpha12)))

        alpha21 = alpha21 + two_pi / 2.0
        if ( alpha21 < 0.0 ) :
                alpha21 = alpha21 + two_pi
        if ( alpha21 > two_pi ) :
                alpha21 = alpha21 - two_pi

        phi2       = phi2       * 45.0 / piD4
        lembda2    = lembda2    * 45.0 / piD4
        alpha21    = alpha21    * 45.0 / piD4

        return phi2,  lembda2,  alpha21 

def lame(E=120,po_ratio=0.30):
    K=E/(3*(1-2*po_ratio))
    lembda=(3*K*(3*K-E))/(9*K-E)
    mu=(3*K*E)/(9*K-E)
    return lembda,mu
def get_random_alphaNumeric_string(stringLength=4):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))

def laplacian2D(N):
            
            diag=np.ones([N*N])
            mat=sp.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
            I=sp.eye(N)
            return sp.kron(I,mat,format='csr')+sp.kron(mat,I)
def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def cut_file(filename,
             dict_types,
             limites=(10, -90),
             colname='Latitud'):
## Get all columns in file that coincide with dict_types. limites
## are the intervals in which we get data accordint to colname
    norte = limites[0]
    sur = limites[1]
    data = []
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        # extract headers
        for row in reader:
            final_types = {key:dict_types.get(key, str) for key in row.keys()}
            new_row = {key: final_types.get(key)(value) for key, value in row.items()}
            value = new_row.get(colname, 0)
            if sur <= value and value <= norte :
                data.append(new_row)
    return data

def find_real(lista, value, esp=0.01):
    ## find respective longitude values that corresponds to 
    ## latitude limit
    for index, elem in enumerate(lista):
        if elem-esp<=value and value<=elem+esp:
            return index, elem

def find_nearest(array,value):
        if type(array) is not np.ndarray:
            array = np.asarray(array)
        idx = np.abs(array-value).argmin()
        return idx

