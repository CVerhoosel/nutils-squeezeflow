from nutils import cli, types, export
import treelog, numpy, pandas

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')
_    = numpy.newaxis

def main(mu0: unit['Pa*s'], muinf: unit['Pa*s'], tcr: unit['s'], n: float, R0: unit['m'], h0: unit['m'], F: unit['N'], T: unit['s'], m: int, npicard: int, ntarget: int, tol: float, s: float, Dtmax: unit['s'], relax: float):
    '''
    Radial squeeze flow of a non-newtonian fluid
    .. arguments::
        mu0 [1.Pa*s]
            Viscosity at zero shear rate
        muinf [0.001Pa*s]
            Viscosity at infinite shear rate
        tcr [0.89s]
            Newton/Power law cross over time scale
        n [0.5]
            Power index
        R0 [8.cm]
            Initial radius of fluid domain
        h0 [0.5mm]
            Initial semi-height of fluid domain
        F [5.N]
            Loading
        T [30s]
            Final time
        m [1000]
            Number of (radial) integration segments
        s [0.1]
            Initial time step scaling
        Dtmax [1.s]
            Maximum time step size
        npicard [50]
            Number of Picard iterations
        ntarget [20]
            Target number of Picard iterations
        relax [0.25]
            Picard relation parameter
        tol [0.0001]
            Tolerance for Picard iterations
    '''

    # initialization
    t = 0
    h = h0
    r = numpy.linspace(0,R0,m+1)
    p = 2*F/(numpy.pi*R0**2)*(1-(r/R0)**2)

    # post processing
    pp = PostProcessing(mu0, muinf, tcr, n)

    # initial time step based on lowest possible viscosity solution
    Δt = s*(3*numpy.pi*muinf*R0**4)/(8*F*h0**2)

    # differentiate the pressure
    dpdr = differentiate(p, r)

    with treelog.context('solving for t={:4.2e} [s]', t/unit('s')) as printtime:
        while t <= T:

            # print the time step
            printtime(t/unit('s'))

            converged = numpy.zeros(shape=dpdr.shape, dtype=bool)
            dpdr0     = dpdr.copy()
            with treelog.iter.plain('picard', range(npicard)) as iterations:
                for iteration in iterations:

                    # compute the interface positions
                    w1 = numpy.minimum(h, (mu0/tcr)*(1/abs(dpdr)))
                    w2 = numpy.minimum(h, (mu0/tcr)*(1/abs(dpdr))*(muinf/mu0)**(n/(n-1)))

                    # compute the diffusion coefficients
                    C1 = (1/mu0)*((mu0/muinf)*w2**2*w1-(mu0/muinf)*h**2*w1-(2/3)*w1**3) + 2*(tcr**(1-n)/mu0)**(1/n)*abs(dpdr)**((1-n)/n)*(n/(n+1))*(w1**((n+1)/n)-w2**((n+1)/n))*w1
                    C2 = 2*(tcr**(1-n)/mu0)**(1/n)*abs(dpdr)**((1-n)/n)*(n/(n+1))*(-(n+1)/(2*n+1)*w2**((2*n+1)/n)-(n/(2*n+1))*w1**((2*n+1)/n)+w2**((n+1)/n)*w1) + (1/muinf)*(w2**2-h**2)*(w2-w1)
                    C3 = (1/muinf)*(-(2/3)*h**3-(1/3)*w2**3+h**2*w2)
                    C  = -(C1+C2+C3)

                    rc   = 0.5*(r[:-1]+r[1:])
                    dr   = r[1:]-r[:-1]
                    f    = (rc/C*dr)[::-1].cumsum()[::-1]
                    hdot = -F/(2*numpy.pi*(rc*f*dr).sum())

                    ddpdr  = rc*hdot/C-dpdr
                    dpdr  += (1-relax)*ddpdr
                    Δdpdr  = dpdr-dpdr0

                    err = abs(ddpdr/Δdpdr)
                    converged = (err<tol)
                    treelog.info(f'{converged.sum()}/{converged.size} converged, max error={numpy.max(err):4.3e}')
                    if all(converged):
                        break

            if all(converged):
                # plot the converged solution
                pp.plot(t, r, dpdr, h, w1, w2)

                # increment the time step
                Δt *= ntarget/iteration
                Δt  = min(Δt, Dtmax)
                h  += hdot*Δt
                R   = R0*numpy.sqrt(h0/h)
                r   = numpy.linspace(0,R,m+1)
                t  += Δt
            else:
                treelog.info(f'picard solver did not converge in {npicard} iterations for {converged.size-converged.sum()} point(s)')
                Δt *= ntarget/iteration
                dpdr = dpdr0.copy()

    # return the time series data frame
    return pp.df

class PostProcessing:

    def __init__(self, mu0, muinf, tcr, n, nz=100):
        self.df = pandas.DataFrame({'t': [], 'h': [], 'R': []})
        self.mu0   = mu0
        self.muinf = muinf
        self.tcr   = tcr
        self.n     = n
        self.nz    = nz

    def plot(self, t, r, dpdr, h, w1, w2):

        rc = 0.5*(r[:-1]+r[1:])
        dr = differentiate(r)

        p = numpy.append(0,numpy.cumsum(dpdr*dr))-(dpdr*dr).sum()

        with export.mplfigure('p.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='p [Pa]')
            ax.plot(r / unit('mm'), p / unit('Pa'))
            ax.grid()

        with export.mplfigure('dpdr.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='dp/dr [Pa/mm]')
            ax.plot(rc / unit('mm'), dpdr / (unit('Pa')/unit('mm')))
            ax.grid()

        # contour plots
        z  = numpy.linspace(0,h,self.nz)
        zc = 0.5*(z[:-1]+z[1:])
        rm, zm = numpy.meshgrid(rc, zc)

        region = (zm>w1).astype(int)+(zm>w2).astype(int)

        γ1 = abs(1/self.mu0*dpdr[_,:]*zm)
        γ2 = abs(dpdr[_,:]*(self.tcr**(1-self.n)/self.mu0)**(1/self.n)*abs(dpdr[_,:])**((1-self.n)/self.n)*(zm**(1/self.n)))
        γ3 = abs(1/self.muinf*dpdr[_,:]*zm)
        γ  = numpy.choose(region, [γ1, γ2, γ3])

        μ1 = self.mu0*numpy.ones_like(γ1)
        μ2 = self.mu0*(self.tcr*γ2)**(self.n-1)
        μ3 = self.muinf*numpy.ones_like(γ2)
        μ  = numpy.choose(region, [μ1, μ2, μ3])

        with export.mplfigure('regions.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]', ylim=(0, 1.1*h))
            ax.plot(rc / unit('mm'), h * numpy.ones_like(rc), label='$h$')
            ax.plot(rc / unit('mm'), w1, label='$w_1$')
            ax.plot(rc / unit('mm'), w2, label='$w_2$')
            ax.grid()
            ax.legend()

        with export.mplfigure('gamma.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]')
            img = ax.contourf(rm/unit('mm'), zm/unit('mm'), γ*unit('s'))
            fig.colorbar(img, label='γ [1/s]')

        with export.mplfigure('viscosity.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]')
            img = ax.contourf(rm/unit('mm'), zm/unit('mm'), μ/unit('Pa*s'))
            fig.colorbar(img, label='μ [Pa s]')

        # time plots
        R = r[-1]
        self.df = pandas.concat([self.df, pandas.DataFrame({'t': [t], 'h': [h], 'R': [R]})], ignore_index=True)
        
        if self.df.shape[0]>0:
            with export.mplfigure('R.png') as fig:
                ax = fig.add_subplot(111, xlabel='t [s]', ylabel='R [mm]')
                ax.plot(self.df['t'] / unit('s'), self.df['R'] / unit('mm'), '.-')
                ax.grid()

            with export.mplfigure('h.png') as fig:
                ax = fig.add_subplot(111, xlabel='$t$ [s]', ylabel='$h$ [mm]')
                ax.plot(self.df['t'] / unit('s'), self.df['h'] / unit('mm'), '.-')
                ax.grid()
                ax.legend()

        if self.df.shape[0]>1:
            hdot = differentiate(self.df['h'].to_numpy(), self.df['t'].to_numpy())
            with export.mplfigure('hdot.png') as fig:
                ax = fig.add_subplot(111, xlabel='t [s]', ylabel='hdot [mm/s]')
                ax.plot(self.df['t'][:-1] / unit('s'), hdot / (unit('mm')/unit('s')), '.-')
                ax.grid()

        # save data frame to file
        with treelog.userfile('timeseries.csv', 'w') as f:
            self.df.to_csv(f, index=False)

def differentiate(f, x=None):
    if x is None:
        return f[1:]-f[:-1]
    return (f[1:]-f[:-1])/(x[1:]-x[:-1])

cli.run(main)
