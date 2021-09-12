import numpy as np
from scipy.linalg import solve_banded

F = 96485 # C/mol, Faraday constant    
R = 8.315 # J/mol K, Gas constant

class Macro:

    def __init__(self, wf):
        self.E = wf.E
        self.t = wf.t
        self.dt = self.t[1]
        self.dT = self.dt/self.t[-1]
        self.nT = np.size(self.t)
        self.maxT = 1

    def Emech(self, n=1, c=1e-6, D=1e-5, Ageo=1, E0=0, ks=1e8, alpha=0.5, CdlRu=[0,0], T=298):
        self.n = n
        self.c = c
        self.D = D
        self.Ageo = Ageo
        self.E0 = E0
        self.ks = ks
        self.alpha = alpha
        self.Cdl = CdlRu[0]
        self.Ru = CdlRu[1]
        self.T = T
        self.FRT = F/(R*self.T)
        self.delta = np.sqrt(self.D*self.t[-1])
        self.K0 = self.ks*self.delta/self.D

    def BI(self, dX=2e-3):
        maxX = 6*np.sqrt(self.maxT)
        dX = dX
        nX = int(maxX/dX)
        X = np.linspace(0,maxX,nX)
        lamb = self.dT/dX**2

        # Thomas coefficients
        a = -lamb
        b = 1 + 2*lamb
        g = -lamb

        C = np.ones([self.nT,nX]) # Initial condition for C
        self.V = np.zeros(self.nT+1)
        iF = np.zeros(self.nT)

        # Constructing ab to use in solve_banded:
        ab = np.zeros([3,nX])
        ab[0,2:] = g
        ab[1,:] = b
        ab[2,:-2] = a
        ab[1,0] = 1
        ab[1,-1] = 1

        # Initial condition for V
        self.V[0] = self.E[0]

        #%% Simulation
        for k in range(0,self.nT-1):
            if self.Cdl:
                eps = self.FRT*(self.V[k] - self.E0)
            else:
                eps = -self.FRT*(self.E[k] - self.E0)
            
            # Butler-Volmer:
            b0 = -(1 +dX*self.K0*(np.exp((1-self.alpha)*eps) + np.exp(-self.alpha*eps)))
            g0 = 1
            
            # Updating ab with the new values
            ab[0,1] = g0
            ab[1,0] = b0
            
            # Boundary conditions:
            C[k,0] = -dX*self.K0*np.exp(-self.alpha*eps)
            C[k,-1] = 1
            
            C[k+1,:] = solve_banded((1,1), ab, C[k,:])
            
            # Obtaining faradaic current and solving voltage drop
            iF[k] = self.n*F*self.Ageo*self.D*self.c*(-C[k+1,2] + 4*C[k+1,1] - 3*C[k+1,0])/(2*dX*self.delta)
            if self.Cdl:
                self.V[k+1] = (self.V[k] + (self.t[1]/self.Cdl)*(self.E[k]/self.Ru -iF[k]))/(1 + self.t[1]/(self.Cdl*self.Ru))
        
        if self.Cdl:
            self.i = (self.E-self.V[:-1])/self.Ru
        else:
            self.i = -iF

        # Denormalising:
        self.cR = C*self.c
        self.cO = self.c - self.cR
        self.x = X*self.delta

