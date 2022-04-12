from nutils import cli, mesh, function, solver, export, types, testing
import numpy, treelog, typing, pandas

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')
_ = numpy.newaxis

def main(muinf:unit['Pa*s'], mu0:unit['Pa*s'], l:unit['s'], nu:float, R0:unit['m'], h0:unit['m'], F:unit['N'], T:unit['s'], m:int, urdegree:int, uzdegree:int, npicard:int, tol:float, nt:int, ratio:int, plotnewtonian:bool):

    '''
    Radial squeeze flow of a Carreau fluid

    .. arguments::

       muinf [0.5Pa*s]
         Viscosity at infinite shear rate

       mu0 [1.0Pa*s]
         Viscosity at zero shear rate

       l [0.005s]
         Relaxation time

       nu [2.]
         Power index

       R0 [2cm]
         Initial radius of fluid domain

       h0 [1mm]
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

       npicard [1000]
         Number of Picard iterations

       tol [0.001]
         Tolerance for Picard iterations

       urdegree [3]
         Velocity order in radial direction

       uzdegree [10]
         Velocity order in thickness direction

       plotnewtonian [False]
         Flag to plot the Newtoian reference solution

    .. presets::

       newtonian
         nu=1.
         l=0s
         mu0=1Pa*s
         muinf=1Pa*s
         plotnewtonian=True

       carreau
         nu=0.05
         muinf=0Pa*s
         mu0=200Pa*s
         l=0.2s
         F=5.0N
         R0=1.0cm
         h0=0.5mm
         plotnewtonian=True
         uzdegree=10
         T=5s

       experiment
         nu=0.043
         l=0.2s
         mu0=10Pa*s
         muinf=0.001Pa*s
         F=5.690N
         R0=10.6mm
         h0=0.45mm
         nt=500
         T=600s
         ratio=1000000

 '''

    assert uzdegree > 1
    pdegree  = urdegree-1

    ρdomain, ρgeom = mesh.line(numpy.linspace(0, 1, m+1), bnames=('inner','outer'), space='R')
    ζdomain, ζgeom = mesh.line(numpy.linspace(-1/2, 1/2, 2), bnames=('bottom','top'), space='H')

    domain = ρdomain*ζdomain

    ns = function.Namespace()
    pp = PostProcessing(ρdomain, ζdomain, ns, h0, R0, plotnewtonian)

    ns.δh = '(?h - ?h0) / ?Δt'
    ns.ρ  = ρgeom
    ns.ζ  = ζgeom
    ns.r  = 'ρ ?Rpicard'
    ns.z  = 'ζ ?hpicard'
    ns.x  = function.stack([ns.r,ns.z])

    urbasis = ρdomain.basis('std', degree=urdegree)
    uzbasis = ζdomain.basis('std', degree=uzdegree)

    ns.ubasis  = function.ravel(urbasis[:,_]*uzbasis[_,:], axis=0)
    ns.u       = function.dotarg('u', ns.ubasis)
    ns.upicard = function.dotarg('upicard', ns.ubasis)

    ns.pbasis = ρdomain.basis('std', degree=pdegree)
    ns.p = function.dotarg('p', ns.pbasis)

    usqr    = domain.boundary['inner,top,bottom'].integral('u^2 d:x' @ ns, degree=2*uzdegree)
    ucons   = solver.optimize('u', usqr, droptol=1e-15, arguments={'hpicard':h0, 'Rpicard':R0})
    upicard = numpy.zeros_like(ucons)
    Rpicard = R0
    hpicard = h0

    ns.π        = numpy.pi
    ns.F        = F
    ns.λ        = l
    ns.ν        = nu
    ns.μinf     = muinf
    ns.μ0       = mu0
    ns.δγ       = 'u_,1'
    ns.δΓ       = 'sqrt(2 (u_,0)^2 + (u_,1)^2)' 
    ns.δΓpicard = 'sqrt(2 (upicard_,0)^2 + (upicard_,1)^2)' 
    ns.μeff     = 'μinf + (μ0 - μinf) (1 + (λ δΓpicard)^2)^((ν - 1) / 2)'
    ns.τ        = 'μeff δγ'

    sample = ζdomain.sample('gauss',2*uzdegree)*ρdomain.sample('gauss',2*urdegree)

    ures = sample.integral('((ubasis_n r)_,1 τ - (ubasis_n r)_,0 p) d:x' @ ns)
    pres = sample.integral('pbasis_n (-(u r)_,0 - r (δh / ?hpicard)) d:x' @ ns)
    hres = domain.boundary['top'].integral('((F / (?Rpicard^2)) - π p) r d:x' @ ns, degree=2*urdegree)

    Δtsteps = numpy.power(ratio**(1/(nt-1)),range(nt))
    Δtsteps = (T/Δtsteps.sum())*Δtsteps
    treelog.user(f'Min Δt: {numpy.min(Δtsteps):4.2e}s')
    treelog.user(f'Max Δt: {numpy.max(Δtsteps):4.2e}s')
    with treelog.iter.plain('timestep',range(nt)) as steps:
        for step in steps:

            Δt = Δtsteps[step]

            with treelog.iter.plain('picard',range(npicard)) as iterations:
                for iteration in iterations:
                    args = dict(upicard=upicard,h0=h0,hpicard=hpicard,Rpicard=Rpicard,Δt=Δt,t=Δtsteps[:step+1].sum())
                    state = solver.solve_linear(('u', 'p', 'h'), (ures, pres, hres), constrain=dict(u=ucons), arguments=args)
                    state.update(args)
                    state['R'] = numpy.sqrt(hpicard/state['h'])*Rpicard

                    area, μerror= domain.integrate(['r d:x', '((μeff - μinf - (μ0 - μinf) (1 + (λ δΓ)^2)^((ν - 1) / 2)) / μ0)^2 r d:x'] @ ns, arguments=state, degree=2*uzdegree)
                    μerror = numpy.sqrt(μerror/area)
                    Rerror = abs(Rpicard-state['R'])/Rpicard

                    treelog.user(f'viscosiy error = {μerror:4.3e}')
                    treelog.user(f'radius error = {Rerror:4.3e}')

                    upicard = state['u']
                    hpicard = state['h']
                    Rpicard = state['R']

                    if μerror < tol and Rerror < tol:
                        break
                else:
                    raise RuntimeError(f'Picard solver did not converge in {npicard} iterations')

            # plot the results
            pp.plot(state)

            # set initial values for th next step
            h0 = state['h']

    # return the time series data frame
    return pp.df

class PostProcessing:

    def __init__(self, ρdomain, ζdomain, ns, h0, R0, plotnewtonian, npointsρ=6, npointsζ=20):
        self.interior   = ρdomain.sample('bezier', npointsρ)*ζdomain.sample('bezier', npointsζ)
        self.centerline = ζdomain.sample('uniform',1)*ρdomain.sample('bezier',npointsρ)
        self.profile    = ρdomain.sample('uniform',1)*ζdomain.sample('bezier',npointsζ)
        self.npointsζ   = npointsζ
        self.ns         = ns
        self.df         = pandas.DataFrame({'t':[0.], 'h':[h0], 'R':[R0]})
        self.plotana    = plotnewtonian

    def plot(self, state):

        ns = self.ns.copy_() # copy namespace so that we don't modify the calling argument

        # field plots
        x, μeff, u, p, δΓ = self.interior.eval(['x_i', 'μeff', 'u', 'p', 'δΓ'] @ ns, **state)

        with export.mplfigure('shearrate.png') as fig:
          ax = fig.subplots(1,1)
          im = ax.tripcolor(x[:,0], x[:,1], self.interior.tri, δΓ, shading='gouraud', cmap='jet')
          fig.colorbar(im)

        with export.mplfigure('viscosity.png') as fig:
          ax = fig.subplots(1,1)
          im = ax.tripcolor(x[:,0], x[:,1], self.interior.tri, μeff, shading='gouraud', cmap='jet')
          im.set_clim(ns.μinf.eval(),ns.μ0.eval())
          fig.colorbar(im)

        with export.mplfigure('carreau.png') as fig:
          ax = fig.subplots(1,1)
          im = ax.loglog(δΓ, μeff, '.', alpha=0.1)
          ax.set_xlabel(r'$\dot{\gamma}$')
          ax.set_ylabel(r'$\mu_{eff}$')
          ax.set_ylim(ns.μinf.eval(),ns.μ0.eval())
          ax.grid()

        with export.mplfigure('velocity.png') as fig:
          ax = fig.subplots(1,1)
          im = ax.tripcolor(x[:,0], x[:,1], self.interior.tri, u, shading='gouraud', cmap='jet')
          fig.colorbar(im)

        with export.mplfigure('pressure.png') as fig:
          ax = fig.subplots(1,1)
          im = ax.tripcolor(x[:,0], x[:,1], self.interior.tri, p, shading='gouraud', cmap='jet')
          fig.colorbar(im)

        # centerline plot
        r, u, p = self.centerline.eval(['r', 'u', 'p'] @ ns, **state)

        with export.mplfigure('centerline.png') as fig:
          ax = fig.subplots(2,1)
          ax[0].plot(r, u, label='FE')
          ax[0].set_ylabel('u')
          ax[0].legend()
          ax[0].grid()
          ax[1].plot(r, p, label='FE')
          ax[1].set_ylabel('p')
          ax[1].set_xlabel('r')
          ax[1].legend()
          ax[1].grid()

        # profile plot
        z, u, p = self.profile.eval(['z', 'u', 'p'] @ ns, **state)

        with export.mplfigure('profiles.png') as fig:
          ax = fig.subplots(2,1)
          ax[0].plot(u.reshape(-1,self.npointsζ).T, z.reshape(-1,self.npointsζ).T)
          ax[1].plot(p.reshape(-1,self.npointsζ).T, z.reshape(-1,self.npointsζ).T)

        # time plots
        self.df = self.df.append({'t':state['t'], 'h':state['h'], 'R':state['R']}, ignore_index=True)

        t_ana = numpy.linspace(0,self.df['t'].max(),100)
        h0 = self.df['h'][0]
        R0 = self.df['R'][0]

        F  = ns.F.eval()
        μ0 = ns.μ0.eval()
        h_ana = h0*(1 + (8*F*h0**2)/(3*numpy.pi*μ0*(R0**4))*t_ana)**(-1/4)
        R_ana = R0*numpy.sqrt(h0/h_ana)

        with export.mplfigure('radius.png') as fig:
            ax = fig.subplots(1,1)
            ax.plot(self.df['t'], self.df['R'], '.-', label='FE')
            if self.plotana:
                ax.plot(t_ana, R_ana, ':', label='analytical (Newtonian)')
            ax.set_xlabel('t')
            ax.set_ylabel('R')
            ax.grid()
            ax.legend()

        with export.mplfigure('height.png') as fig:
            ax = fig.subplots(1,1)
            ax.plot(self.df['t'], self.df['h'], '.-', label='FE')
            if self.plotana:
                ax.plot(t_ana, h_ana, ':', label='analytical (Newtonian)')
            ax.set_xlabel('t')
            ax.set_ylabel('h')
            ax.grid()
            ax.legend()

        # save data frame to file
        with treelog.userfile('timeseries.csv', 'w') as f:
            self.df.to_csv(f, index=False)

if __name__ == '__main__':
    cli.run(main)

# Testing
class test(testing.TestCase):

    @testing.requires('matplotlib')
    def test_newtonian(self):
        df = main(muinf=unit('1Pa*s'), mu0=unit('1Pa*s'), l=unit('0s'), nu=1., R0=unit('2cm'), h0=unit('1mm'), F=unit('1N'), T=unit('30s'), m=2, urdegree=3, uzdegree=2, npicard=5, tol=1e-3, nt=40, ratio=100, plotnewtonian=True)
        with self.subTest('radius'):
            self.assertAlmostEqual(df['R'].values[-1], 0.037395942122931285, places=6)
