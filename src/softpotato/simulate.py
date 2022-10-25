import numpy as np

## Electrochemistry constants
F = 96485 # C/mol, Faraday constant    
R = 8.315 # J/mol K, Gas constant
T = 298 # K, Temperature
FRT = F/(R*T)


class E:
    def __init__(self, wf, n=1, A=0.0314, E0=0, cOb=0, cRb=1e-6, 
                 DO=1e-5, DR=1e-5, k0=1e8, alpha=0.5):
        self.wf = wf
        self.n = n
        self.A = A
        self.E0 = E0
        self.cOb = cOb
        self.cRb = cRb
        self.DO = DO
        self.DR = DR
        self.k0 = k0
        self.alpha = alpha
        self.DOR = DO/DR

    def grid(self):
        self.nT = np.size(self.wf.t)
        self.dT = 1/self.nT
        self.lamb = 0.45
        self.Xmax = 6*np.sqrt(self.nT*self.lamb)
        self.dX = np.sqrt(self.dT/self.lamb)
        self.nX = int(self.Xmax/self.dX)

        ## Discretisation of variables and initialisation
        if self.cRb == 0: # In case only O present in solution
            self.CR = np.zeros([self.nT,self.nX])
            self.CO = np.ones([self.nT,self.nX])
        else:
            self.CR = np.ones([self.nT,self.nX])
            self.CO = np.ones([self.nT,self.nX])*self.cOb/self.cRb

        self.X = np.linspace(0,self.Xmax,self.nX) # Discretisation of distance
        self.eps = (self.wf.E-self.E0)*self.n*FRT # adimensional potential waveform
        self.delta = np.sqrt(self.DR*self.wf.t[-1]) # cm, diffusion layer thickness
        self.K0 = self.k0*self.delta/self.DR # Normalised standard rate constant


    def run(self):
        self.grid()
        for k in range(1,self.nT):
            # Boundary condition, Butler-Volmer:
            CR1kb = self.CR[k-1,1]
            CO1kb = self.CO[k-1,1]
            self.CR[k,0] = (CR1kb + self.dX*self.K0*np.exp(
                            -self.alpha*self.eps[k])*(CO1kb + CR1kb/self.DOR))/(
                       1 + self.dX*self.K0*(np.exp((1-self.alpha)*self.eps[k])+\
                       np.exp(-self.alpha*self.eps[k])/self.DOR))
            self.CO[k,0] = CO1kb + (CR1kb - self.CR[k,0])/self.DOR

            self.CR[k,1:-1] = self.CR[k-1,1:-1] + self.lamb*(self.CR[k-1,2:]\
                            - 2*self.CR[k-1,1:-1] + self.CR[k-1,:-2])
            self.CO[k,1:-1] = self.CO[k-1,1:-1] + self.lamb*(self.CO[k-1, 2:]\
                            - 2*self.CO[k-1, 1:-1] + self.CO[k-1,:-2])

        # Denormalising:
        if self.cRb:
            I = -self.CR[:,2] + 4*self.CR[:,1] - 3*self.CR[:,0]
            D = self.DR
            c = self.cRb
        else: # In case only O present in solution
            I = self.CO[:,2] - 4*self.CO[:,1] + 3*self.CO[:,0]
            D = self.DO
            c = self.cOb
        self.i = self.n*F*self.A*D*c*I/(2*self.dX*self.delta)

        self.cR = self.CR*self.cRb
        self.cO = self.CO*self.cOb
        self.x = self.X*self.delta

            


if __name__ == '__main__':
    import softpotato as sp
    print('Running from main')

    # Potential waveform:
    wf = sp.technique.Sweep(Eini=-0.5, Efin=0.5, sr=0.1)

    # Electrochemical parameters:
    n = 1
    A = 0.0314
    E0 = 0
    cOb = 0
    cRb = 1e-6
    DO = 3.25e-5
    DR = 3.25e-5
    k0 = 1e8
    alpha = 0.5

    # Simulation:
    sim = E(wf, n, A, E0, cOb, cRb, DO, DR, k0, alpha)
    sim.run()
    
    # Plotting
    sp.plotting.plot(wf.t, wf.E, xlab='$t$ / s', ylab='$E$ / V', fig=1, show=0)
    sp.plotting.plot(wf.E, sim.i*1e6, ylab='$i$ / $\mu$A', fig=2, show=1)

