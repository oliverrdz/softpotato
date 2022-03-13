import numpy as np

F = 96485 # C/mol
R = 8.3145 # J/(mol K)


class Macro:
    '''
    '''

    def __init__(self, n=1, A=1, C=1e-6, D=1e-5, T=298):
        self.n = n
        self.A = A
        self.C = C
        self.D = D
        self.T = 298

    def Cottrell(self, t):
        return self.n*F*self.A*self.C*np.sqrt(self.D/(np.pi*t))

    def RandlesSevcik(self, sr):
        i = 0.4463*self.n*F*self.A*self.C*np.sqrt(self.n*F*sr*self.D/(R*self.T))
        return i


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
    
    def __init__(self, a=5e-4, n=1, DO=1e-5, DR=1e-5, cOb=1e-6, cRb=0, E0=0, 
                 k0=1e8, alpha=0.5, T=298, noise=0):
        self.a = a
        self.n = n
        self.DO = DO
        self.DR = DR
        self.cOb = cOb
        self.cRb = cRb
        self.E0 = E0
        self.k0 = k0
        self.alpha = alpha
        self.T = T
        self.FRT = F/(R*T)
        self.noise = noise
        if self.cOb > 0 and self.cRb == 0:
            print('Reduction')
            self.C = self.cOb
            self.D = self.DO
            self.alphaf = -alpha
            self.alphab = 1-alpha
            self.DODR = DO/DR
            self.sign = -1
        elif self.cOb == 0 and self.cRb > 0:
            print('Oxidation')
            self.C = self.cRb
            self.D = self.DR
            self.DODR = DR/DO
            self.alphaf = -1+alpha
            self.alphab = alpha
            self.sign = 1
        else:
            print('Error. Either CO or CR should be zero.') 
        self.iLim = self.sign*4*n*F*self.D*self.C*self.a

    def LSV(self, E):
        kf = self.k0*np.exp(self.alphaf*self.n*self.FRT*(E-self.E0))
        kb = self.k0*np.exp(self.alphab*self.n*self.FRT*(E-self.E0))
        if self.sign == -1:
            k = kf
        else:
            k = kb
        theta=1+self.DODR*np.exp(-self.sign*self.n*F*(E-self.E0)/(R*self.T))
        kappa = np.pi*k*self.a/(4*self.D)
        i = (self.iLim/theta)/(1 + 
            (np.pi/(kappa*theta))*((2*kappa*theta+
            3*np.pi)/(4*kappa*theta+3*np.pi**2)))
        return i

    def CA2(self, t, Es=-1):
        self.t = t
        self.Es = Es
        fO = self.fun(self.DO)
        fR = self.fun(self.DR)
        Theta = 1 + (self.DO*fO/(self.DR*fR))*np.exp(self.n*F*(self.Es-self.E0)/(R*self.T))
        kappa = (self.k0*self.a/(self.DO*fO))*np.exp(-self.alpha*self.n*F*(self.Es-self.E0)/(R*self.T))
        ca = -(np.pi*self.n*F*self.DO*self.CO*self.a*fO/Theta)/(1+(np.pi/(kappa*Theta))*((2*kappa*Theta+3*np.pi)/(4*kappa*Theta+3*np.pi**2)))
        return ca + np.random.normal(size=self.t.size, scale=self.noise)

    def fun2(self, D):
        s = D*self.t/self.a**2
        f1 = 1/np.sqrt(np.pi*s) + 1 + np.sqrt(s/(4*np.pi)) - 3*s/25 + 3*s**(3/2)/226
        f2 = 4/np.pi + 8/np.sqrt(s*np.pi**5) + 25/(2792*s**(3/2)) - 1/(3880*s**(5/2)) - 1/(4500*s**(7/2))
        # Fancy if statement using boolean operations:
        return (s<1.281)*f1 + (s>=1.281)*f2

if __name__ == '__main__':
    import plotting
    print('Running from main')
    ox = 1
    if ox:
        disc = MicroDisc(cOb=0, cRb=1e-6, k0=1e-3, alpha=0.5)
    else:
        disc = MicroDisc(cOb=1e-6, cRb=0, k0=1e-3, alpha=0.5)
    print(disc.iLim)
    E = np.linspace(1,-1)
    i = disc.LSV(E)
    plotting.plot(E,i,show=1)

