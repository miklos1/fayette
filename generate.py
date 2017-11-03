import os

import numpy
import pandas

assembly = pandas.read_csv("assembly.csv")
assembly["rate"] = assembly.num_dofs / assembly.parloop_time

matvec = pandas.read_csv("matvec.csv")
matvec["rate"] = matvec.num_dofs / matvec.matvec_time

matfree = pandas.read_csv("matfree.csv")
matfree["rate"] = matfree.num_dofs / matfree.matmult_time

print('Files read.')

outdir = "data"
os.makedirs(outdir, exist_ok=True)


mutate = {"poisson": "poisson",
          "hyperelasticity": "hyperelastic",
          "curl_curl": "curlcurl",
          "stokes_momentum": "stokes_momentum"}


def curve(dataset, problem, config, exp, prefix=""):
    name = problem
    if config == "base":
        mode = "coffee"
    elif config == "spectral":
        mode = "spectral"
    elif config == "underintegration":
        problem = {"poisson": "poisson_gll"}[problem]
        mode = "spectral"
    elif config == "spmv":
        pass
    else:
        assert False, "Unexpected configuration!"

    filtered = dataset.loc[lambda r: r.problem == problem]
    if config != "spmv":
        filtered = filtered.loc[lambda r: r.tsfc_mode == mode]
    num_procs, = set(filtered["num_procs"])

    series = filtered.groupby(["degree"]).mean()["rate"]
    series.to_csv("%s/%s%s_%s.csv" % (outdir, prefix, mutate[name], config), header=True)

    array = numpy.array(list(series.to_dict().items()))
    x = array[:, 0]
    y = array[:, 1]
    logC = numpy.log(y) - numpy.log(x**3 / (x+1)**exp)
    rho = logC.std() / logC.mean()
    if rho > 0.1:
        print(problem, config, 'rho =', rho)
    C = numpy.exp(logC.mean())
    return C, int(numpy.floor(x.min())), int(numpy.ceil(x.max()))


def linear(problem):
    with open("%s/%s.dat" % (outdir, mutate[problem]), 'w') as f:
        print('C a b', file=f)
        C, a, b = curve(matvec, problem, "spmv", 6)
        # print(C, a, b, file=f)
        C, a, b = curve(matfree, problem, "base", 6)
        print(C, a, b, file=f)
        C, a, b = curve(matfree, problem, "spectral", 4)
        print(C, a, b, file=f)


def bilinear(problem):
    with open("%s/bi%s.dat" % (outdir, mutate[problem]), 'w') as f:
        print('C a b', file=f)
        C, a, b = curve(assembly, problem, "base", 9, prefix="bi")
        print(C, a, b, file=f)
        C, a, b = curve(assembly, problem, "spectral", 7, prefix="bi")
        print(C, a, b, file=f)


def bilinear_poisson():
    with open("%s/bipoisson.dat" % (outdir,), 'w') as f:
        print('C a b', file=f)
        C, a, b = curve(assembly, "poisson", "base", 9, prefix="bi")
        print(C, a, b, file=f)
        C, a, b = curve(assembly, "poisson", "spectral", 7, prefix="bi")
        print(C, a, b, file=f)
        C, a, b = curve(assembly, "poisson", "underintegration", 6, prefix="bi")
        print(C, a, b, file=f)


def bilinear_stokes_momentum():
    curve(assembly, "stokes_momentum", "base", 9)
    curve(assembly, "stokes_momentum", "spectral", 9)


bilinear_stokes_momentum()
bilinear_poisson()
bilinear("hyperelasticity")
bilinear("curl_curl")

linear("poisson")
linear("hyperelasticity")
linear("curl_curl")
