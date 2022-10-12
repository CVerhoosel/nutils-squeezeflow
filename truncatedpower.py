from nutils import cli, mesh, function, solver, export, types
import treelog, numpy, typing, pandas

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')

def main(mu0:unit['Pa*s'], tcr:unit['s'], nu:float, R0:unit['m'], H0:unit['m'], F:unit['N'], T:unit['s'], m:int, degree:int, npicard:int, tol:float, nt:int, ratio:int, plotnewtonian:bool):

    '''
    Radial squeeze flow of a Carreau fluid

    .. arguments::

       mu0 [1.0Pa*s]
         Viscosity at zero shear rate

       tcr [0.1s]
         Newton/Power law cross over time scale

       nu [0.5]
         Power index

       R0 [2cm]
         Initial radius of fluid domain

       H0 [1mm]
         Initial height of fluid domain

       F [1N]
         Loading

       T [30s]
         Final time

       m [16]
         Number of (radial) elements

       ratio [100]
         Ratio between largest and smallest time step

       nt [40]
         Number of time steps

       npicard [50]
         Number of Picard iterations

       tol [0.001]
         Tolerance for Picard iterations

       degree [3]
         Pressure order in radial direction

       plotnewtonian [False]
         Flag to plot the Newtoian reference solution

    .. presets::

       newtonian
         nu=1.
         mu0=1Pa*s
         plotnewtonian=True

       carreau
         nu=0.05
         mu0=200Pa*s
         F=5.0N
         R0=1.0cm
         h0=0.5mm
         plotnewtonian=True
         T=5s

 '''

    assert 0. < nu <= 1.0

    # initialize namespace with constants
    ns   = function.Namespace()
    ns.π = numpy.pi

    # physical parameters
    ns.F   = F
    ns.μ0  = mu0
    ns.tcr = tcr
    ns.K   = mu0*tcr**(nu-1)
    ns.ν   = nu

    # plot the constitutive relation
    with export.mplfigure('viscosity.png') as fig:
        ax = fig.add_subplot(111, xlabel='$\dot{\gamma}$ [1/s]', ylabel='$\mu$ [Pa·s]')
        γdot = (1/ns.tcr.eval())*numpy.power(10,numpy.linspace(-3,3,100))
        mu   = numpy.minimum(mu0*numpy.ones_like(γdot), ns.K.eval()*γdot**(nu-1))
        ax.loglog(γdot/unit('1/s'), mu/unit('Pa*s'))
        ax.grid()

    # domain and geometry definition
    domain, ρgeom = mesh.line(numpy.linspace(0, 1, m+1), bnames=('inner','outer'), space='R')
    h0            = H0/2
    ns.V          = numpy.pi*H0*R0**2
    ns.h          = '?h'
    ns.h0         = '?h0'
    ns.hpicard    = '?hpicard'
    ns.R          = 'sqrt(V / (2 π h))'
    ns.R0         = 'sqrt(V / (2 π h0))'
    ns.Rpicard    = 'sqrt(V / (2 π hpicard))'
    ns.ρ          = ρgeom
    ns.r          = 'ρ Rpicard'
    ns.x          = function.stack([ns.r])
    ns.δh         = '(h - h0) / ?Δt'
    ns.wpicard    = 'hpicard' # using mid point works well

    # pressure discretization
    ns.basis = domain.basis('spline', degree)
    ns.p       = 'basis_i ?lhs_i'
    ns.ppicard = 'basis_i ?lhspicard_i'

    ns.wpicard = function.min(ns.hpicard, abs('-(K)^(1 / (1 - ν)) μ0^(ν / (ν - 1)) / ppicard_,0'@ns))
    # ns.wpicard = ns.hpicard / 1

    # flux definition
    ns.Cnewton = '-((2 r) / (3 μ0)) (hpicard)^3'
    ns.Cpower = 0
    # ns.Cnewton = '2 r ((-(wpicard^3) / (3 μ0)) + ((ν / (1 + ν)) (abs(ppicard_,0) / K)^((1 / ν) - 1) (hpicard^((1 / ν) + 1) - wpicard^((1 / ν) + 1)) wpicard))'
    # ns.Cpower  = '2 r ((ν / (1 + ν)) (abs(ppicard_,0) / K)^((1 / ν) - 1) ((((1 + ν) / (1 + 2 ν)) hpicard^((1 / ν) + 2)) - (hpicard^((1 / ν) + 1) wpicard) + ((ν / (1 + 2 ν)) wpicard^((1 / ν) + 2))))'
    ns.Q = '(Cnewton + Cpower) p_,0'
  
    # pressure boundary condition
    sqr = domain.boundary['outer'].integral('p^2'@ns, degree=4)
    cons = solver.optimize('lhs', sqr, droptol=1e-10)

    # definition of the residuals
    resp = domain.integral('(2 δh basis_i r - Q basis_i,0) d:x'@ns, degree=4)
    resh = domain.integral('((F / (Rpicard^2)) - π p) r d:x'@ns, degree=4)

    # initialization
    state = {'h0':h0, 'lhs0':numpy.zeros_like(cons)}
    pp    = PostProcessing(domain, ns(**state), plotnewtonian)

    # time iteration
    Δtsteps = numpy.power(ratio**(1/(nt-1)),range(nt))
    Δtsteps = (T/Δtsteps.sum())*Δtsteps
    treelog.user(f'Min Δt: {numpy.min(Δtsteps):4.2e}s')
    treelog.user(f'Max Δt: {numpy.max(Δtsteps):4.2e}s')
    with treelog.iter.plain('timestep',range(nt)) as steps:
        for step in steps:

            state['Δt']        = Δtsteps[step]
            state['t']         = Δtsteps[:step+1].sum()
            state['hpicard']   = state['h0']
            state['lhspicard'] = state['lhs0']

            with treelog.iter.plain('picard',range(npicard)) as iterations:
                for iteration in iterations:

                    # solve the linear system of equations
                    state = solver.solve_linear(['lhs','h'], [resp,resh], constrain={'lhs':cons}, arguments=state)

		    # check convergence
                    herror = abs(state['hpicard']-state['h'])/abs(state['h0'])
                    treelog.user(f'height error = {herror:4.3e}')
                    if herror < tol:
                        break
            
                    relax = 1.0
                    state['lhspicard'] = (1-relax)*state['lhspicard'] + relax*state['lhs']
                    state['hpicard']   = (1-relax)*state['hpicard'] + relax*state['h']

            # postprocessing
            pp.plot(state)
            
            # set initial conditions for the next time step
            state['lhs0'] = state['lhs']
            state['h0']   = state['h']

    # return the time series data frame
    return pp.df

class PostProcessing:

    def __init__(self, domain, ns, plotnewtonian, npoints=6):
        self.interior   = domain.sample('bezier', npoints)
        self.ns         = ns
        self.df         = pandas.DataFrame({'t':[0.], 'h':[ns.h0.eval()], 'R':[ns.R0.eval()]})
        self.plotana    = plotnewtonian

    def plot(self, state):

        ns = self.ns(**state) # copy namespace so that we don't modify the calling argument

        # plots
        x, p, h, w = self.interior.eval(['r', 'p', 'h', 'wpicard'] @ ns)

        with export.mplfigure('pressure.png') as fig:
          ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='p [Pa]')
          ax.plot(x/unit('mm'), p/unit('Pa'))
          ax.grid()

        with export.mplfigure('height.png') as fig:
          ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]', ylim=(0, 1.1*numpy.max(h)))
          ax.plot(x/unit('mm'), h, label='$h$')
          ax.plot(x/unit('mm'), w, label='$w$')
          ax.grid()
          ax.legend()

        # time plots
        self.df = self.df.append({'t':state['t'], 'h':state['h'], 'R':ns.R.eval()}, ignore_index=True)

        t_ana = numpy.linspace(0,self.df['t'].max(),100)
        h0 = self.df['h'][0]
        R0 = self.df['R'][0]

        F  = ns.F.eval()
        μ0 = ns.μ0.eval()
        h_ana = h0*(1 + (8*F*(2*h0)**2)/(3*numpy.pi*μ0*(R0**4))*t_ana)**(-1/4)
        R_ana = R0*numpy.sqrt(h0/h_ana)

        with export.mplfigure('radius.png') as fig:
            ax = fig.add_subplot(111, xlabel='t [s]', ylabel='R [mm]')
            ax.plot(self.df['t']/unit('s'), self.df['R']/unit('mm'), '.-', label='FE')
            if self.plotana:
                ax.plot(t_ana/unit('s'), R_ana/unit('mm'), ':', label='analytical (Newtonian)')
            ax.grid()
            ax.legend()

        with export.mplfigure('semiheight.png') as fig:
            ax = fig.add_subplot(111, xlabel='$t$ [s]', ylabel='$h$ [mm]')
            ax.plot(self.df['t']/unit('s'), self.df['h']/unit('mm'), '.-', label='$h$')
            if self.plotana:
                ax.plot(t_ana/unit('s'), h_ana/unit('mm'), ':', label='analytical (Newtonian)')
            ax.grid()
            ax.legend()

        # save data frame to file
        with treelog.userfile('timeseries.csv', 'w') as f:
            self.df.to_csv(f, index=False)
  
cli.run(main)
