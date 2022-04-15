import squeezeflow
import json, numpy, treelog
from pathlib import Path
from nutils import cli, export

script_path = Path(__file__).parent.absolute()

def sample(fname='data/input.json',
           N=2500):

    # load the input file
    defaults, header, data = load_input(fname)

    # create the output directory
    outdir = script_path/'data/output'
    if not outdir.is_dir():
        Path.mkdir(outdir)

    # loop over the samples
    with treelog.iter.plain('sample', data[:N,:]) as samples:
        for sample in samples:

            ID = int(sample[0])

            outfile      = outdir/f'{ID:04d}.csv'
            outfile_lock = outdir/f'{ID:04d}.csv.lock'

            # check whether the sample was already treated
            if outfile.is_file() or outfile_lock.is_file() or (outdir/f'{ID:04d}.csv.fail').is_file():
                if outfile.is_file():
                    treelog.user(f'Sample {ID} already treated')
                elif outfile_lock.is_file():
                    treelog.user(f'Sample {ID} currently locked')
                elif (outdir/f'{ID:04d}.csv.fail').is_file():
                    treelog.user(f'Sample {ID} failed before')
                continue
            else:
                treelog.user(f'Running sample {ID}')
                outfile_lock.touch()

            # get the input arguments
            kwargs = {arg:(val if not unit else squeezeflow.unit(val+unit)) for arg, val, unit in defaults}
            kwargs.update({arg:(val if not unit else squeezeflow.unit(f'{val:16.12f}'+unit)) for val, (arg, unit) in zip(sample[1:], header[1:])})

            # run the simulation
            try:
                df = squeezeflow.main(**kwargs)
                with open(outfile, 'w') as f:
                    df.to_csv(f, index=False)
                    outfile_lock.unlink()
            except:
                outfile_lock.rename(outfile_lock.with_suffix('.fail'))

def inspect(fname='data/input.json',
            experiment='data/experiment.csv'):

    # load the input file
    defaults, header, data = load_input(fname)

    # collect results
    outdir = script_path/'data/output'

    data  = []
    fails = 0
    for path in outdir.iterdir():
        if path.is_file() and path.suffix=='.csv':
            data.append(numpy.loadtxt(path, delimiter=',', skiprows=1)[numpy.newaxis,:,:])
        elif path.is_file() and path.suffix=='.fail':
            fails += 1

    data = numpy.concatenate(data, axis=0)

    N = data.shape[0]
    treelog.user(f'Valid samples: {N}')
    treelog.user(f'Failed samples: {fails}')

    μ = numpy.mean(data, axis=0)
    σ = numpy.std(data, axis=0)

    # load the experiment
    exp = numpy.loadtxt(script_path/experiment, skiprows=1, delimiter=',')

    with export.mplfigure('radius.png') as fig:
        ax = fig.subplots(1,1)
        ax.plot(data[:,:,0].T, data[:,:,-1].T, 'r', alpha=0.3)
        ax.errorbar(exp[:,0], exp[:,1], exp[:,2], fmt='g.', ecolor='g', label='exp')
        ax.plot(μ[:,0], μ[:,-1], 'b', label='μ')
        ax.plot(μ[:,0], numpy.array([μ[:,-1]+σ[:,-1], μ[:,-1]-σ[:,-1]]).T, 'b--', label='μ±σ')
        ax.set_xlabel('t')
        ax.set_ylabel('R')
        ax.set_title(f'N = {N}')
        ax.grid()
        ax.legend()

def load_input(fname):

    # load the json with the column information
    with open(script_path/fname) as f:
        d = json.load(f)

    # load the input file
    fin = script_path/f'data/{d["name"]}'
    data = numpy.loadtxt(fin, skiprows=1, delimiter=',')

    assert data.shape[1]==len(d['header'])

    treelog.user(f'Input file N = {data.shape[0]}')

    return d['defaults'], d["header"], data

cli.choose(sample, inspect)
