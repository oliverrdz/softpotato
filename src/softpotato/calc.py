import numpy as np

F = 96485 # C/mol
R = 8.3145 # J/(mol K)


class MicroDisc:
    '''
    ----------
    Description:
    Calculates the current of a microdisc electrode
    All parameters are given a default value

    ----------
    Parameters:
    a:      cm, electrode radius [5e-4 cm]
    n:      number of electrons [1]
    DO:     cm2/s, diffusion coefficient of species O [1e-5 cm2/s]
    DR:     cm2/s, diffusioen coefficient of species R [1e-5 cm2/s]
    CO:     mol/cm3, bulk concentration of species O [1e-6 mol/cm3]
    E0:     V, standard potential [0 V]
    k0:     cm/s, standard rate constant [1e8 cm/s]
    alpha:  transfer coefficient [0.5]
    T:      K, temperature [298 K]
    noise:  A, random normal distributed noise


    ----------
    Returns:
    self.iLim:  A, limiting current

    ----------
    Methods:
    LSV(E):
        Parameters:
        E:      V, required, numpy array to calculate the voltammogram
        Returns:
        self.lsv: A, numpy array with the currents calculated at potentials E
    CA():
        Parameters:
        t:      s, numpy array with the times to calculate the current
        E:      V, potential for the step
        Returns:
        self.ca: A, numpy array with the currents calculated at times t


    ----------
    Examples:
    > import softpotato as sp
    > E = sp.technique.Sweep()
    > disc = sp.calc.MicroDisc(a=12.5e-4) # Changing only the electrode radius
    > lsv = disc.voltammetry(E)
    > print(disc.iLim)
    > print(lsv)

    '''
    
    def __init__(self, a=5e-4, n=1, DO=1e-5, DR=1e-6, CO=1e-6, E0=0, k0=1e8, alpha=0.5, T=298, noise=0):
        self.a = a
        self.n = n
        self.DO = DO
        self.DR = DR
        self.CO = CO
        self.E0 = E0
        self.k0 = k0
        self.alpha = alpha
        self.T = T
        self.noise = noise
        self.iLim = 4*self.n*F*self.DO*self.CO*self.a

    def LSV(self, E):
        self.E = E
        self.kappa0 = np.pi*self.k0*self.a/(4*self.DO)
        self.Theta = 1 + (self.DO/self.DR)*np.exp(self.n*F*(self.E-self.E0)/(R*self.T))
        self.kappa = self.kappa0*np.exp(-self.alpha*self.n*F*(self.E-self.E0)/(R*self.T))
        lsv = -(self.iLim/self.Theta)/(1+(np.pi/(self.kappa*self.Theta))*((2*self.kappa*self.Theta+3*np.pi)/(4*self.kappa*self.Theta+3*np.pi**2)))
        return lsv + np.random.normal(size=self.E.size, scale=self.noise)

    def CA(self, t, Es=-1):
        self.t = t
        self.Es = Es
        fO = self.fun(self.DO)
        fR = self.fun(self.DR)
        Theta = 1 + (self.DO*fO/(self.DR*fR))*np.exp(self.n*F*(self.Es-self.E0)/(R*self.T))
        kappa = (self.k0*self.a/(self.DO*fO))*np.exp(-self.alpha*self.n*F*(self.Es-self.E0)/(R*self.T))
        ca = -(np.pi*self.n*F*self.DO*self.CO*self.a*fO/Theta)/(1+(np.pi/(kappa*Theta))*((2*kappa*Theta+3*np.pi)/(4*kappa*Theta+3*np.pi**2)))
        return ca + np.random.normal(size=self.t.size, scale=self.noise)

    def fun(self, D):
        s = D*self.t/self.a**2
        f1 = 1/np.sqrt(np.pi*s) + 1 + np.sqrt(s/(4*np.pi)) - 3*s/25 + 3*s**(3/2)/226
        f2 = 4/np.pi + 8/np.sqrt(s*np.pi**5) + 25/(2792*s**(3/2)) - 1/(3880*s**(5/2)) - 1/(4500*s**(7/2))
        # Fancy if statement using boolean operations:
        return (s<1.281)*f1 + (s>=1.281)*f2
