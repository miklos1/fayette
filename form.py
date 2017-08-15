from __future__ import absolute_import, division, print_function
from six.moves import range

import FIAT
import finat

from firedrake import (Constant, FiniteElement, Function,
                       FunctionSpace, Identity, TestFunction,
                       TrialFunction, VectorFunctionSpace, curl,
                       derivative, diff, dot, dx, grad, inner, tr,
                       variable)


# Gauss-Lobatto-Legendre quadrature rule

def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())


def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result


# Weak forms

def mass(mesh, degree):
    V = FunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    return inner(u, v)*dx


def mass_dq(mesh, degree):
    V = FunctionSpace(mesh, FiniteElement('DQ', mesh.ufl_cell(),
                                          degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return inner(u, v)*dx


def mass_gll(mesh, degree):
    V = FunctionSpace(mesh, FiniteElement('Q', mesh.ufl_cell(),
                                          degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    dim = mesh.topological_dimension()
    finat_rule = gauss_lobatto_legendre_cube_rule(dim, degree)
    return inner(u, v)*dx(rule=finat_rule)


def mass_vec(mesh, degree):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    return inner(u, v)*dx


def poisson(mesh, degree):
    V = FunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(grad(u), grad(v))*dx


def poisson_gll(mesh, degree):
    V = FunctionSpace(mesh, FiniteElement('Q', mesh.ufl_cell(),
                                          degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    dim = mesh.topological_dimension()
    finat_rule = gauss_lobatto_legendre_cube_rule(dim, degree)
    return dot(grad(u), grad(v))*dx(rule=finat_rule)


def helmholtz(mesh, degree):
    V = FunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    return (dot(grad(u), grad(v)) + u*v)*dx


def stokes_momentum(mesh, degree):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    return inner(grad(u), grad(v))*dx


def elasticity(mesh, degree):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    def eps(u):
        return (grad(u) + grad(u).T) / 2
    return inner(eps(v), eps(u))*dx

    return inner(grad(u), grad(v))*dx


def hyperelasticity(mesh, degree):
    V = VectorFunctionSpace(mesh, 'Q', degree)
    v = TestFunction(V)
    du = TrialFunction(V)  # Incremental displacement
    u = Function(V)        # Displacement from previous iteration
    B = Function(V)        # Body force per unit mass
    # Kinematics
    I = Identity(mesh.topological_dimension())
    F = I + grad(u)        # Deformation gradient
    C = F.T*F              # Right Cauchy-Green tensor
    E = (C - I)/2          # Euler-Lagrange strain tensor
    E = variable(E)
    # Material constants
    mu = Constant(1.0)     # Lame's constants
    lmbda = Constant(0.001)
    # Strain energy function (material model)
    psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)
    S = diff(psi, E)       # Second Piola-Kirchhoff stress tensor
    PK = F*S               # First Piola-Kirchoff stress tensor
    # Variational problem
    return derivative((inner(PK, grad(v)) - inner(B, v))*dx, u, du)


def curl_curl(mesh, degree):
    cell = mesh.ufl_cell()
    if cell.cellname() in ['interval * interval', 'quadrilateral']:
        hcurl_element = FiniteElement('RTCE', cell, degree)
    elif cell.cellname() == 'quadrilateral * interval':
        hcurl_element = FiniteElement('NCE', cell, degree)
    V = FunctionSpace(mesh, hcurl_element)
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(curl(u), curl(v))*dx
