from __future__ import absolute_import, division, print_function
from six.moves import range

import os
import sys

import pandas

from argparse import ArgumentParser
from numpy import cbrt, floor, sqrt
from mpi4py import MPI

from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, ExtrudedMesh, UnitSquareMesh, assemble

import form


parser = ArgumentParser(description="""Profile assembly""", add_help=False)

parser.add_argument("--mode", action="store", default="spectral",
                    help="Which TSFC mode to use?")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?  Default is to append.")

parser.add_argument("--help", action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)

assembly_path = os.path.abspath("assembly.csv")
matvec_path = os.path.abspath("matvec.csv")

num_matvecs = 40


PETSc.Log.begin()

parloop_event = PETSc.Log.Event("ParLoopExecute")
assemble_event = PETSc.Log.Event("AssembleMat")
matmult_event = PETSc.Log.Event("MatMult")


simplex_range = list(range(1, 5))
if args.mode == "coffee":
    cube_range = list(range(1, 10))
elif args.mode == "spectral":
    cube_range = list(range(1, 16))


test_cases = [
    (form.stokes_momentum, False),
    (form.elasticity, False),
    (form.poisson, True),
    (form.mass_gll, True),
    (form.poisson_gll, True),
    (form.hyperelasticity, True),
    (form.curl_curl, True),
]
for problem, tensor in test_cases:
    formname = problem.__name__
    if args.mode == "spectral" and problem == form.curl_curl:
        degrees = list(range(1, 12))
    elif tensor:
        degrees = cube_range
    else:
        degrees = simplex_range
    for degree in degrees:
        cellname = 'cube' if tensor else 'simplex'
        PETSc.Sys.Print("%s: %s, degree=%d" % (formname, cellname, degree))
        num_cells = COMM_WORLD.size * max(1e5 / degree**4, 1e8 / (degree + 1)**9)
        num_cells = max(COMM_WORLD.size, min((1 << 27) / (degree + 1)**6, num_cells))
        h = int(floor(cbrt(num_cells / COMM_WORLD.size)))
        w = int(floor(sqrt(num_cells / h)))
        d = int(round(num_cells / (w * h)))
        num_cells = w * d * h

        mesh = ExtrudedMesh(UnitSquareMesh(w, d, quadrilateral=tensor), h)
        comm = mesh.comm
        J = problem(mesh, int(degree))

        # Warmup and allocate
        A = assemble(J, form_compiler_parameters={'mode': args.mode})
        A.force_evaluation()
        Ap = A.petscmat
        x, y = Ap.createVecs()
        assert x.size == y.size
        num_dofs = x.size
        Ap.mult(x, y)
        stage = PETSc.Log.Stage("%s(%d) %s" % (formname, degree, cellname))
        with stage:
            with assemble_event:
                assemble(J, form_compiler_parameters={'mode': args.mode}, tensor=A)
                A.force_evaluation()
                Ap = A.petscmat
            for _ in range(num_matvecs):
                Ap.mult(x, y)

            parloop = parloop_event.getPerfInfo()
            assembly = assemble_event.getPerfInfo()
            matmult = matmult_event.getPerfInfo()
            assert parloop["count"] == 1
            assert assembly["count"] == 1
            assert matmult["count"] == num_matvecs
            parloop_time = comm.allreduce(parloop["time"], op=MPI.SUM) / comm.size
            assemble_time = comm.allreduce(assembly["time"], op=MPI.SUM) / comm.size
            matmult_time = comm.allreduce(matmult["time"], op=MPI.SUM) / (comm.size * num_matvecs)

            assembly_overhead = (1 - parloop_time / assemble_time)
            PETSc.Sys.Print("Assembly overhead: %.1f%%" % (assembly_overhead * 100,))

        if COMM_WORLD.rank == 0:
            if args.overwrite:
                mode = "w"
                header = True
            else:
                mode = "a"
                header = not os.path.exists(assembly_path)

            data = {"num_cells": num_cells,
                    "num_dofs": num_dofs,
                    "num_procs": comm.size,
                    "tsfc_mode": args.mode,
                    "problem": formname,
                    "cell_type": cellname,
                    "degree": degree,
                    "assemble_time": assemble_time,
                    "parloop_time": parloop_time}
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(assembly_path, index=False, mode=mode, header=header)

            if not args.overwrite:
                header = not os.path.exists(matvec_path)

            data = {"num_cells": num_cells,
                    "num_dofs": num_dofs,
                    "num_procs": comm.size,
                    "tsfc_mode": args.mode,
                    "problem": formname,
                    "cell_type": cellname,
                    "degree": degree,
                    "matvec_time": matmult_time}
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(matvec_path, index=False, mode=mode, header=header)
