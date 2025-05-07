import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import os 
from pathlib import Path 


class Limb_Darkening:
    """
    A class for computing limb darkening effects and transit light curves
    using non-linear 4-coefficient models.

    Attributes
    ----------
    data_dict : dict
        Dictionary storing limb darkening coefficients from a data file.
    logg_range, teff_range, m_H_range : numpy.ndarray
        Arrays containing available grid values for log(g), Teff, and [M/H].

    Methods
    -------
    get_coeff(logg, Teff, m_H):
        Retrieve the closest-match limb darkening coefficients (a1–a4)
        for given stellar parameters.

    F_e(p, z):
        Calculate geometric flux fraction based on eclipse geometry
        without considering limb darkening.

    I(r, coeff):
        Compute intensity at a normalized radial distance 'r' from the stellar center
        using the non-linear limb darkening model with given coefficients.

    F(p, z, coeff):
        Calculate normalized light curve flux considering limb darkening
        for a planet of radius ratio 'p' at projected distances 'z'.
    """
    def __init__(self):
        dir_name = os.path.dirname(__file__)
        addre = Path(dir_name)/'LD_coeff.tsv'
        data = pd.read_csv(addre,sep = '|') 
        self.data_dict = {}
        self.logg_range = []
        self.teff_range = []
        self.m_H_range = []
        for i,r in data.iterrows():
            self.data_dict[r['Coeff'].strip(' ')+' '+str(int(r['logg']*2))+' '+str(r['Teff'])+' '+str(int(r['log[M/H]']*10))] = r['V']
            if not int(r['logg']*2) in self.logg_range:
                self.logg_range.append(int(r['logg']*2))
            if not r['Teff'] in self.teff_range:
                self.teff_range.append(r['Teff'])
            if not int(r['log[M/H]']*10) in self.m_H_range:
                self.m_H_range.append(int(r['log[M/H]']*10))
        self.logg_range = np.array(self.logg_range)
        self.teff_range = np.array(self.teff_range)
        self.m_H_range = np.array(self.m_H_range)
    def get_coeff(self,logg,Teff,m_H):
        """
        Retrieve the closest-match limb darkening coefficients (a1–a4) for given stellar parameters.
        """
        logg = logg*2
        m_H = 10*m_H
        str_key = ' '+str(self.logg_range[np.argmin(np.abs(self.logg_range-logg))])+' '+str(self.teff_range[np.argmin(np.abs(self.teff_range-Teff))])+' '+str(self.m_H_range[np.argmin(np.abs(self.m_H_range-m_H))])
        return [self.data_dict['a1'+str_key],self.data_dict['a2'+str_key],self.data_dict['a3'+str_key],self.data_dict['a4'+str_key]]
    def F_e(self,p,z):
        """ 
        Calculate geometric flux fraction based on eclipse geometry
        without considering limb darkening.
        """
        k1 = np.arccos((1-p**2+z**2)/2/z)
        k0 = np.arccos((p**2+z**2-1)/2/p/z)
        return 1-np.where(1+p<z,0,np.where((np.abs(1-p)<z)&(z<=1+p),1/np.pi*(p**2*k0+k1-np.sqrt((4*z**2-(1+z**2-p**2)**2)/4)),np.where(z<=1-p,p**2,1)))
    def I(self,r,coeff):
        """ 
        Compute intensity at a normalized radial distance 'r' from the stellar center
        using the non-linear limb darkening model with given coefficients.
        """
        mu=np.sqrt(1-r**2)
        return 1-coeff[0]*(1-mu**0.5)-coeff[1]*(1-mu**1)-coeff[2]*(1-mu**1.5)-coeff[3]*(1-mu**2)
    def F(self,p,z,coeff):
        """ 
        Calculate normalized light curve flux considering limb darkening
        for a planet of radius ratio 'p' at projected distances 'z'.        
        """
        z = z.reshape(1,-1)
        r_list = np.linspace(10**-6,1,300).reshape(-1,1)
        
        Ir = self.I(r_list,coeff)
        Fer2 = self.F_e(p/r_list,z/r_list)*r_list**2
        dFer2 = Fer2[1:,:]-Fer2[:-1,:]
        B = np.sum(dFer2*Ir[1:,:],axis = 0)
        A = np.sum(r_list*2*Ir)*(r_list[2,0]-r_list[1,0])
        return B/A

class Generator:
    """
    A class to generate synthetic planetary transits and corresponding light curves
    based on stellar and orbital parameters, including limb darkening effects.

    Attributes
    ----------
    period_planet : numpy.ndarray
        Logarithmically spaced planet period grid (days).
    mass_list, radius_planet : numpy.ndarray
        Planet mass and radius grid (in Earth units).
    occurance_rate : numpy.ndarray
        2D array of planet occurrence rates from Kepler-like population studies.
    ld : Limb_Darkening
        An instance of the Limb_Darkening class to compute stellar intensity profiles.

    Methods
    -------
    generate_planets(multi_planet):
        Randomly generate one or more planets based on the occurrence rate map.

    generate_S2(R=1, n=1):
        Generate n random unit vectors uniformly distributed over the sphere.

    generate_coord_trans_mat_schimidt(n=1):
        Generate n random 3D rotation matrices to simulate orbital orientation.

    light_curve(star, planet, orbit_dir, show=False):
        Compute the exact transit light curve for a given star and planet,
        accounting for geometry, orbital orientation, and limb darkening.

    generate_obs_lc(light_curve_exact, accuracy, noise):
        Simulate observational noise (white noise) on a generated light curve.

    Run(time, star, planet):
        Wrapper function to generate a complete synthetic light curve given time,
        stellar, and planetary inputs. Handles full orbit simulation.
    """
    def __init__(self, star_path = 'Gaia_area_par.csv', time_path = 'result/obs_time.npy', time_mode = 'load'):
        # day
        self.period_planet = np.array([0.4,0.4**0.5,1,4**(1/3),4**(2/3)]+list(np.array([0.4,0.4**0.5,1,4**(1/3),4**(2/3)])*10)+list(np.array([0.4,0.4**0.5,1,4**(1/3),4**(2/3)])*100)+[400])
        #Earth radius
        self.mass_list = 2**np.arange(-3,13, dtype=np.float64)#*const.M_earth
        self.period_list  = 6.25/8*np.power(2,np.arange(14))
        self.radius_planet = np.array([20,8,4,2,1])
        # From 2103.02127 Fig.3
        self.occurance_rate = np.array(
        [
            [0.009,0.012,0.022,0.059,0.21,0.18,0.15,0.20,0.17,0.23,0.52,0.44,0.13,0.12,0.17],
            [0.009,0.012,0.016,0.05,0.07,0.19,0.15,0.43,0.45,0.66,0.58,1.5,1.3,2.4,1.7],
            [0.14,0.12,0.33,0.89,0.37,1.25,2.52,4.3,6.4,7.2,8.1,6.9,6.2,7.5,6.9],
            [0.039,0.158,0.20,0.54,1.45,2.97,4.1,5.0,4.6,4.1,4.2,3.1,4.2,4.9,10]
        ]
        )/100
        

        self.ld=Limb_Darkening()
    def generate_planets(self,multi_planet):
        """ 
        Randomly generate one or more planets based on the occurrence rate map.
        """
        planets = []
        for radius_index in range(len(self.radius_planet)-1):
            for period_index in range(len(self.period_planet)-1):
                if np.random.rand() < self.occurance_rate[radius_index][period_index]: #generate this planet
                    portion = np.random.rand()
                    planet_period = np.exp(np.log(self.period_planet[period_index])*portion + np.log(self.period_planet[period_index+1])*(1-portion))
                    portion = np.random.rand()
                    planet_radius = np.exp(np.log(self.radius_planet[radius_index])*portion + np.log(self.radius_planet[radius_index+1])*(1-portion))
                    phase = 2*np.pi*np.random.rand()
                    planets.append({'period':planet_period,'radius':planet_radius,'initial_phase':phase})  
                    if not multi_planet:
                        return planets
        return planets         
    def generate_S2(self,R=1,n=1):
        """ 
        Generate n random unit vectors uniformly distributed over the sphere.
        """
        coords=[]
        while len(coords)<n:
            while(1):
                coord=np.random.rand((3))
                coord=coord*2-1
                if np.sum(coord**2)<1:
                    break
            D=np.sqrt(np.sum(coord**2))
            coord=coord*R/D
            coords.append(coord)
        return np.array(coords)
    def generate_coord_trans_mat_schimidt(self,n=1):
        """ 
        Generate n random 3D rotation matrices to simulate orbital orientation.
        """
        x_axis = self.generate_S2(n=n)
        y_axis0 = self.generate_S2(n=n)
        y_axis_o = y_axis0-x_axis*(np.sum(x_axis*y_axis0,axis=1).reshape(-1,1))
        y_axis = y_axis_o/(np.sqrt(np.sum(y_axis_o*y_axis_o,axis=1)).reshape(-1,1))
        z_axis = np.cross(x_axis,y_axis)
        rot_mat = np.array([x_axis.T,y_axis.T,z_axis.T])
        return rot_mat
    
    def light_curve(self,star,planet,orbit_dir,show = False):
        """ 
        Compute the exact transit light curve for a given star and planet,
        accounting for geometry, orbital orientation, and limb darkening.
        """
        def get_z(pos,star):
            return np.where(pos[0,:]<0,100,((pos[1,:]**2+pos[2,:]**2)**0.5*u.m/star['radius']/u.R_sun).to(1))
        lc = np.ones(len(self.time))
        sub_part = np.zeros(len(self.time))
        coeff = self.ld.get_coeff(star['logg'],star['teff'],star['mh'])
        num_transit_planet = 0

        phase = planet['initial_phase']+self.time.jd*2*np.pi/planet['period']
        semi_major_axis = (((planet['period']*u.day)**2/4/np.pi**2*const.G*star['mass']*const.M_sun)**(1/3)).to(u.m)
        pos_planet_orbital_plane = np.array([semi_major_axis*np.cos(phase),semi_major_axis*np.sin(phase),u.m*np.zeros(phase.shape)]).reshape(3,-1)
        pos_planet_3d = orbit_dir.dot(pos_planet_orbital_plane)
        
        p = float((planet['radius']*const.R_earth/star['radius']/const.R_sun).to(1))
        z = get_z(pos_planet_3d,star)
        if np.min(z)>1+p:
            return num_transit_planet,lc
        num_transit_planet +=1
        lc_this_planet = self.ld.F(p,np.array(z),coeff)

        sub_part += (1-lc_this_planet)
        lc = lc - sub_part

        return num_transit_planet,lc

    def generate_obs_lc(self,light_curve_exact,accuracy,noise):
        """ 
        Simulate observational noise (white noise) on a generated light curve.
        """
        if noise == 'white':
            flucation =  accuracy * np.random.randn(len(light_curve_exact))
            res = flucation + light_curve_exact
            return res
    def Run(self,time,star = {'teff':6000,'mh':0,'logg':1.5,'radius':1,'mass':1},planet = {'radius':1,'period':365,'initial_phase':1.3*np.pi}):
        """ 
        Wrapper function to generate a complete synthetic light curve given time,
        stellar, and planetary inputs. Handles full orbit simulation.
        """
        self.time = Time(time,format='jd',scale='utc')

        while 1: 
            
            print('try')
            orbit_dir = np.eye(3)
            num_transit_planet,light_curve_exact = self.light_curve(star,planet,orbit_dir)
            
            
            light_curve_exact = light_curve_exact+(1-np.max(light_curve_exact))
            return light_curve_exact
  