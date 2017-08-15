from __future__ import absolute_import, division, print_function
from six.moves import filter, intern, map, range, zip

from functools import reduce

from numpy import cbrt, floor, sqrt

from firedrake.petsc import PETSc
from firedrake import ExtrudedMesh, UnitSquareMesh, assemble

import form


num_matvecs = 20


PETSc.Log.begin()

parloop_event = PETSc.Log.Event("ParLoopExecute")
assemble_event = PETSc.Log.Event("AssembleMat")
matmult_event = PETSc.Log.Event("MatMult")


problems = [form.mass, form.mass_dq, form.mass_gll, form.mass_vec,
            form.poisson, form.poisson_gll, form.helmholtz,
            form.stokes_momentum, form.elasticity,
            form.hyperelasticity, form.curl_curl]
for problem in problems:
    name = problem.__name__
    PETSc.Sys.Print(name)
    for degree in range(1, 4):
        num_cells = max(1, 1e8 / (degree + 1)**7)
        w = int(floor(cbrt(num_cells)))
        d = int(floor(sqrt(num_cells / w)))
        h = int(round(num_cells / (w * d)))
        num_cells = w * d * h

        PETSc.Sys.Print("degree = {}: num_cells = {}".format(degree, num_cells))
        mesh = ExtrudedMesh(UnitSquareMesh(w, d, quadrilateral=True), h)
        J = problem(mesh, degree)

        for typ in ["aij", "matfree"]:
            # Warmup and allocate
            A = assemble(J, mat_type=typ)
            A.force_evaluation()
            Ap = A.petscmat
            x, y = Ap.createVecs()
            Ap.mult(x, y)
            stage = PETSc.Log.Stage("%s(%d) %s matrix" % (name, degree, typ))
            with stage:
                with assemble_event:
                    assemble(J, mat_type=typ, tensor=A)
                    A.force_evaluation()
                    Ap = A.petscmat
                for _ in range(num_matvecs):
                    Ap.mult(x, y)

                parloop = parloop_event.getPerfInfo()
                assembly = assemble_event.getPerfInfo()
                matmult = matmult_event.getPerfInfo()
                matmult_time = matmult["time"] / num_matvecs
                assemble_time = assembly["time"]
                print(typ, assemble_time, matmult_time, parloop["time"] / parloop["count"], parloop["count"])
