import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
from classy import Class

class MultiCorr(object):
    """
        This class computes the relativistic 2PCF and power-spectra multipoles
        
        !!! -> Fiducial cosmology must be directly altered in this file! <- !!!
            This makes the code more flexible, so you can compute the 
            multipoles at any redshift and with all the parameters you 
            wish.
        
        OBS: non-Gaussian corrections have not been implemented here yet.
             Once it is functional, this class will depend on halo tools.
        
        No variables are needed to initialize the class  
    
    """
    
    init = False
    
    def __init__(self):

        print '[Initializing MultiCorr]: Relativistic 2PCF and power-spectra multipoles\n'
        
        # Fiducial cosmology
        c = 299792.458
        self.c = c
        h = 0.67556
        self.h = h
        n_s = 0.9619
        self.n_s = n_s
        A_s = 2.215e-9
        self.A_s = A_s
        omega_cdm = 0.12038
        self.omega_cdm = omega_cdm
        omega_b = 0.022032
        self.omega_b = omega_b
        omega_k = 0.0
        self.omega_k = omega_k
        k_pivot = 0.05
        self.k_pivot = k_pivot
        N_ur = 3.046
        self.N_ur = N_ur
        N_ncdm = 0.0
        self.N_ncdm = N_ncdm
        T_cmb = 2.7255
        self.T_cmb = T_cmb
        Omega_m = (omega_cdm + omega_b)/h**2
        self.Omega_m = Omega_m
        w = -1.0
        self.w = w
        
        print '(Pre-defined cosmology)'
        print 'h:', h
        print 'n_s:', n_s
        print 'A_s:', A_s
        print 'N_ur:', N_ur
        print 'N_ncdm:', N_ncdm
        print 'T_cmb:', T_cmb
        print 'Omega_m:', Omega_m
        print 'w:', w 
        print '\n'
        
        self.kvec = np.logspace(-7., np.log10(500), 1000)
        self.k_ = np.logspace(-3,2,2000)
        
        # Linear power-spectrum
        class_settings = {'output': 'mPk', 
                  'lensing': 'no',
                  'h': h, 
                  'n_s': n_s,
                  'A_s': A_s, 
                  'omega_cdm': omega_cdm, 
                  'omega_b': omega_b,
                  'Omega_k': 0.0,
                  'k_pivot': k_pivot,
                  'z_max_pk': 10.,
                  'N_ur': N_ur,
                  'N_ncdm': N_ncdm, 
                  'T_cmb': T_cmb,
                  'P_k_max_1/Mpc': 500}
        self.pclass = Class()
        self.pclass.set(class_settings)
        self.pclass.compute()
        
        # These are the CLASS functions (has 0.5% accuracy wrt to the ones I implement by hand)
        self.bg = self.pclass.get_background()
        self.H = interp1d(self.bg['z'],(c/h)*self.bg['H [1/Mpc]'])
        self.comov = interp1d(self.bg['z'],h*self.bg['comov. dist.'])
        self.fz = interp1d(self.bg['z'],self.bg['gr.fac. f'])
        
        # Non-linear power-spectrum
        class_settings_nl = {'output': 'mPk', 
                          'lensing': 'no',
                          'h': h, 
                          'n_s': n_s,
                          'A_s': A_s, 
                          'omega_cdm': omega_cdm, 
                          'omega_b': omega_b,
                          'Omega_k': 0.0,
                          'k_pivot': k_pivot,
                          'z_max_pk': 10.,
                          'non linear': 'halofit',
                          'N_ur': N_ur,
                          'N_ncdm': N_ncdm, 
                          'T_cmb': T_cmb,
                          'P_k_max_1/Mpc': 500}

        
        self.pnlclass = Class()
        self.pnlclass.set(class_settings_nl)
        self.pnlclass.compute()
        
        print '[MultiCorr Initialized]'
        self.init = True
        
    ######################################################################################################
    # Useful functions
    def Om(self, z):        
        """
            Returns the evolving matter density Omega_m(z)
        """
        
        self.z = z
        self.H0Hz2 = pow(1+self.z,3.0)*self.Omega_m + (1-self.Omega_m)
        
        return pow(1+self.z,3.0)*self.Omega_m/self.H0Hz2

    ######################################################################################################
    # RSD

    def Doppler(self, z_, be_):
        """
            Computes the doppler assuming the magnification bias s = 0.
        """
        
        self.z_ = z_
        self.be_ = be_
        self.scale_factor = 1.0/(1.0+self.z_)
        self.Hubble = self.H(self.z_) # (h/Mpc) (Km/s)
        self.R_at_z = self.comov(self.z_) # (Mpc/h)

        return self.be_ - (1.0-1.5*self.Om(self.z_)) - (2.0/(self.scale_factor*(self.Hubble/self.c)*self.R_at_z))

    def DopplerMag(self, z_,be_,sm_):
        """
            Computes the doppler assuming a non-vanishing magnification bias s != 0.
        """
        
        self.z_ = z_
        self.be_ = be_
        self.sm_ = sm_
        self.scale_factor = 1.0/(1.0+self.z_)
        self.Hubble = self.H(self.z_) # (h/Mpc) (Km/s)
        self.R_at_z = self.comov(self.z_) # (Mpc/h)

        return self.be_ - 5*self.sm_ - (1.0-1.5*self.Om(self.z_)) - (1.0/(self.scale_factor*(self.Hubble/self.c)*self.R_at_z))*(2.0 - 5*self.sm_)

    def c0(self, z_,b_alpha_,b_beta_):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        return self.b_alpha_*self.b_beta_ + (1.0/3.0)*self.fz(self.z_)*(self.b_alpha_+self.b_beta_) + (1.0/5.0)*pow(self.fz(self.z_),2.0)

    def c1(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_):
        
        self.z_ = z_
	self.scale_factor = 1.0/(1.0+self.z_)
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.Hubble = self.H(self.z_) # (h/Mpc) (Km/s)
        self.term1 = self.Doppler(self.z_,self.be_alpha_)*(3*self.fz(self.z_)+5*self.b_beta_)
        self.term2 = self.Doppler(self.z_,self.be_beta_)*(3*self.fz(self.z_)+5*self.b_alpha_)
        return (self.fz(self.z_)/5.0)*(self.Hubble/(self.c*self.k_))*(self.term1-self.term2)

    def c1_Mag(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_,s_alpha,s_beta):
        
        self.z_ = z_
	self.scale_factor = 1.0/(1.0+self.z_)
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.s_alpha = s_alpha
        self.s_beta = s_beta
        self.Hubble = self.H(self.z_) # (h/Mpc) (Km/s)
        self.term1 = self.DopplerMag(self.z_,self.be_alpha_,self.s_alpha)*(3*self.fz(self.z_)+5*self.b_beta_)
        self.term2 = self.DopplerMag(self.z_,self.be_beta_,self.s_beta)*(3*self.fz(self.z_)+5*self.b_alpha_)
        
        return (self.fz(self.z_)/5.0)*(self.scale_factor*self.Hubble/(self.c*self.k_))*(self.term1-self.term2)

    def c2(self, z_,b_alpha_,b_beta_):
        
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        return (2.0/3.0)*self.fz(self.z_)*(self.b_alpha_+self.b_beta_)+(4.0/7.0)*pow(self.fz(self.z_),2.0)

    def c3(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_):

        self.z_ = z_
	self.scale_factor = 1.0/(1.0+self.z_)
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.Hubble = self.H(self.z_) # (h/Mpc) (Km/s)
        self.term1 = self.Doppler(self.z_,self.be_alpha_)
        self.term2 = self.Doppler(self.z_,self.be_beta_)
        return (2.0/5.0)*pow(self.fz(self.z_),2.0)*(self.scale_factor*self.Hubble/(self.c*self.k_))*(self.term1-self.term2)

    def c4(self, z_):
        
        self.z_ = z_
        return (8.0/35.0)*pow(self.fz(self.z_),2.0)


    ######################################################################################################
    # Plane-parallel power-spectrum

    ## First order in $\mathcal{H}/k$
    def P0_pp(self, z_,b_alpha_,b_beta_,linear):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.linear = linear
        
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c0(self.z_,self.b_alpha_,self.b_beta_)*self.Pk(k_)

    def P1_pp(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_,linear):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.linear = linear
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c1(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)*self.Pk(self.k_)

    def P1Mag_pp(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_,s_alpha,s_beta,linear):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.s_alpha = s_alpha
        self.s_beta = s_beta
        self.linear = linear
        
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c1_Mag(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_,self.s_alpha,self.s_beta)*self.Pk(self.k_)

    def P2_pp(self, z_,b_alpha_,b_beta_,linear):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.linear = linear
        
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c2(self.z_,self.b_alpha_,self.b_beta_)*self.Pk(self.k_)

    def P3_pp(self, z_,b_alpha_,b_beta_,be_alpha_,be_beta_,linear):
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.linear = linear
        
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c3(self.k_,self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)*self.Pk(self.k_)

    def P4_pp(self, z_,b_alpha_,b_beta_,linear):
        
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.linear = linear
        
        if linear=='linear' or linear == True:
            self.Pk = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else:
            self.Pk = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
        return self.c4(self.z_)*self.Pk(self.k_)

    #################################################################################################################################
    # Correlation function multipoles

    import scipy
    
    def integrand(self, s,ell):
        self.s = s
        self.ell = ell
        return (pow(self.k_,2.)*scipy.special.spherical_jn(self.ell, self.k_*self.s))/(2*pow(np.pi,2.))

    def matter_real_space(self, s,z_,args):
    	self.s = s
    	self.z_ = z_
    	self.args = args
    	
    	self.linear = self.args[0]
    	
    	if self.linear == 'linear' or self.linear == True:
            self.interpol = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else: 
            self.interpol = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))
           
        self.MONO = []

        for i in s:
            self.MONO.append(integrate.simps(self.integrand(i,0)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))

        self.MONO = np.array(self.MONO)

        return self.MONO

    def multipoles(self, s,z_,b_alpha_,b_beta_,be_alpha_,be_beta_,args):
        self.s=s
        self.z_=z_
        self.b_alpha_=b_alpha_
        self.b_beta_=b_beta_
        self.be_alpha_=be_alpha_
        self.be_beta_=be_beta_
        self.args=args

        """
            This gives the non-vanishing ODD multipoles (l=1,3) for the correlation function. 
            Inputs are:
            1) s: a vector of configuration space positions;
            2) z: redshift of calculation;
            3) b_alpha, b_beta: linear biases of tracers alpha and beta (may be the same);
            4) be_alpha, be_beta: evolution biases for the tracers computed at z;
            5) args: a list or array with like 
                     args = ['linear',s_alpha,s_beta] (s = magnification biases)
                  or args = ['linear']
                  or True/False to select linear or non-linear correlation functions;
        """

        if len(self.args) > 1:
            self.linear, self.s_alpha, self.s_beta = self.args[0], self.args[1], self.args[2]
            self.coef_dipo = self.c1_Mag(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_,self.s_alpha,self.s_beta)
        if len(self.args) == 1:
            self.linear = self.args[0]
            self.coef_dipo = self.c1(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)

        self.coef_mono = self.c0(self.z_,self.b_alpha_,self.b_beta_)
        self.coef_quad = self.c2(self.z_,self.b_alpha_,self.b_beta_)
        self.coef_octu = self.c3(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)
        self.coef_hexa = self.c4(self.z_)

        if self.linear == 'linear' or self.linear == True:
            self.interpol = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else: 
            self.interpol = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))

        self.MONO = []
        self.DIPO = []
        self.QUAD = []
        self.OCTU = []
        self.HEXA = []

        for i in s:
            self.MONO.append(integrate.simps(self.coef_mono *self.integrand(i,0)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))
            self.DIPO.append(integrate.simps(self.coef_dipo *self.integrand(i,1)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))
            self.QUAD.append(integrate.simps(self.coef_quad *self.integrand(i,2)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))
            self.OCTU.append(integrate.simps(self.coef_octu *self.integrand(i,3)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))
            self.HEXA.append(integrate.simps(self.coef_hexa *self.integrand(i,4)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))

        self.MONO = pow(1j,0).real * np.array(self.MONO)
        self.DIPO = pow(1j,1).imag * np.array(self.DIPO)
        self.QUAD = pow(1j,2).real * np.array(self.QUAD)
        self.OCTU = pow(1j,3).imag * np.array(self.OCTU)
        self.HEXA = pow(1j,4).real * np.array(self.HEXA)

        return self.MONO, self.DIPO, self.QUAD, self.OCTU, self.HEXA

    def odd_multipoles(self, s,z_,b_alpha_,b_beta_,be_alpha_,be_beta_,args):
        self.s = s
        self.z_ = z_
        self.b_alpha_ = b_alpha_
        self.b_beta_ = b_beta_
        self.be_alpha_ = be_alpha_
        self.be_beta_ = be_beta_
        self.args = args

        """
            This gives the non-vanishing ODD multipoles (l=1,3) for the correlation function. 
            Inputs are:
            1) s: a vector of configuration space positions;
            2) z: redshift of calculation;
            3) b_alpha, b_beta: linear biases of tracers alpha and beta (may be the same);
            4) be_alpha, be_beta: evolution biases for the tracers computed at z;
            5) args: a list or array with like 
                     args = ['linear',s_alpha,s_beta] (s = magnification biases)
                  or args = ['linear']
                  or True/False to select linear or non-linear correlation functions;
        """

        if len(self.args) > 1:
            self.linear, self.s_alpha, self.s_beta = self.args[0], self.args[1], self.args[2]
            self.coef_dipo = self.c1_Mag(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_,self.s_alpha,self.s_beta)
        if len(self.args) == 1:
            self.linear = self.args[0]
            self.coef_dipo = self.c1(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)

        self.coef_octu = self.c3(self.z_,self.b_alpha_,self.b_beta_,self.be_alpha_,self.be_beta_)

        if self.linear == 'linear' or self.linear == True:
            self.interpol = interp1d(self.kvec/self.h,np.array([self.pclass.pk(_k, self.z_) for _k in self.kvec])*(self.h**3.))
        else: 
            self.interpol = interp1d(np.logspace(-5,2,1000)/self.h,np.array([self.pnlclass.pk(_k, self.z_) for _k in np.logspace(-5,2,1000)])*(self.h**3.))

        self.DIPO = []
        self.OCTU = []

        for i in s:
            self.DIPO.append(integrate.simps(self.coef_dipo *self.integrand(i,1)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))
            self.OCTU.append(integrate.simps(self.coef_octu *self.integrand(i,3)*self.interpol(self.k_)*np.exp(-self.k_**5),self.k_))

        self.DIPO = pow(1j,1).imag * np.array(self.DIPO)
        self.OCTU = pow(1j,3).imag * np.array(self.OCTU)

        return self.DIPO, self.OCTU
        
    def help(self):
        print '[Reaching help for MultiCorr]\n'
        print 'Full list of functions and their entries:\n'
        print '(Auxiliary functions)'
        print 'Om(z): Evolving matter density Omega_m(z) - Inputs: z (redshift).'
        print 'Doppler(z, be): Doppler term assuming magnification bias s = 0 - Inputs: z (redshift) and be (evolution bias at z).'
        print 'DopplerMag(z, be,s): Doppler term - Inputs: z (redshift), be (evolution bias at z) and s (magnification bias at z).'
        print 'c0(z,ba,bb): Monopole coefficient - Inputs: z (redshift) and bi (linear bias of tracer i at z).'
        print 'c1(z,ba,bb,be_a,be_b): Dipole coefficient - Inputs: z (redshift), bi (linear bias of tracer i at z) and be_i (evolution bias of tracer i at z).'
        print 'c1_Mag(z,ba,bb,be_a,be_b,sa,sb): Dipole coefficient with magnification - Inputs: z (redshift), bi (linear bias of tracer i at z), be_i (evolution bias of tracer i at z) and si (magnification of tracer i at z).'
        print 'c2(z,ba,bb): Quadrupole coefficient - Inputs: z (redshift) and bi (linear bias of tracer i at z).'
        print 'c3(z,ba,bb,be_a,be_b): Octupole coefficient assuming no magnification - Inputs: z (redshift), bi (linear bias of tracer i at z) and be_i (evolution bias of tracer i at z).'
        print 'c4(z): Hexadecapole coefficient - Inputs: z (redshift).\n'
        print '(Power-spectrum multipoles)'
        print 'P0_pp(z,ba,bb,linear) - linear = True or False'
        print 'P1_pp(z,ba,bb,be_a,be_b,linear) - linear = True or False'
        print 'P1Mag_pp(z,ba,bb,be_a,be_b,sa,sb,linear) - linear = True or False'
        print 'P2_pp(z,ba,bb,linear) - linear = True or False'
        print 'P3_pp(z,ba,bb,be_a,be_b,linear) - linear = True or False'
        print 'P4_pp(z,ba,bb,linear) - linear = True or False\n'
        print '(Correlation function multipoles)'
        print 'integrand(s,ell): returns the integrand of the correlation function.'
        print 'multipoles(s,z,ba,bb,be_a,be_b,args) - args: [True] or [True, sa, sb], si (magnification bias of tracer i)'
        print 'odd_multipoles(s,z,ba,bb,be_a,be_b,args) - args: [True] or [True, sa, sb], si (magnification bias of tracer i)\n'
        print '[End of help]'
