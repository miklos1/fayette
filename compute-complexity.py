import form
import firedrake
import tsfc
from gem import gem
from gem import impero
from functools import reduce, singledispatch
import sympy
from firedrake import COMM_WORLD, ExtrudedMesh, UnitCubeMesh, UnitSquareMesh, assemble


def expression(expr, temporaries, indices, top=False):
    """Walks an impero tree and computes the complexity polynomial

    :arg expr: Impero expression
    :arg temporaries: subexpressions for which temporaries exist.
    :arg indices: dictionary mapping gem indices to names.
    :arg top: ignore the temporary for the root node.
    :returns: Sympy polynomial.
    """
    if not top and expr in temporaries:
        return 0
    else:
        return _expression(expr, temporaries, indices)


@singledispatch
def _expression(expr, temporaries, indices):
    raise AssertionError("cannot compute complexity polynomial for %s" % type(expr))

@_expression.register(gem.Product)
@_expression.register(gem.Sum)
@_expression.register(gem.Division)
def _expression_binary(expr, temporaries, indices):
    return expression(expr.children[0], temporaries, indices) + expression(expr.children[1], temporaries, indices) + 1


@_expression.register(gem.Constant)
@_expression.register(gem.Variable)
@_expression.register(impero.Initialise)
def _expression_noop(expr, temporaries, indices):
    return 0


@_expression.register(gem.Indexed)
@_expression.register(gem.FlexiblyIndexed)
def _expression_indexed(expr, temporaries, indices):
    return expression(expr.children[0], temporaries, indices)


@_expression.register(impero.Block)
def _expression_block(expr, temporaries, indices):
    return reduce(lambda x, y: x+y, map(lambda e: expression(e, temporaries, indices), expr.children))


@_expression.register(gem.IndexSum)
def _expression_block(expr, temporaries, indices):
    return reduce(lambda x, y: x*y, map(lambda i: indices[i], expr.multiindex)) * expression(expr.children[0], temporaries, indices) 


@_expression.register(impero.Evaluate)
def _expression_evaluate(expr, temporaries, indices):
    return expression(expr.expression, temporaries, indices, top=True)


@_expression.register(impero.For)
def _expression_for(expr, temporaries, indices):
    return indices[expr.index] * expression(expr.children[0], temporaries, indices)


@_expression.register(impero.Accumulate)
@_expression.register(impero.ReturnAccumulate)
def _expression_accumulate(expr, temporaries, indices):
    return expression(expr.indexsum.children[0], temporaries, indices, top=True)


@_expression.register(gem.MathFunction)
def _expression_block(expr, temporaries, indices):
    if expr.name == 'abs':
        return expression(expr.children[0], temporaries, indices) + 1
    
    raise AssertionError("cannot compute complexity polynomial for %s" % type(expr))


def complexity(form, parameters):
    impero_kernel, index_names = tsfc.driver.compile_form(form, parameters=parameters)[0]
    c_kernel = tsfc.driver.compile_form(form, parameters=firedrake.parameters['form_compiler'])[0]

    indices={idx: sympy.symbols(name) for idx, name in index_names}

    return expression(impero_kernel.tree, impero_kernel.temporaries, indices, top=True)


m = UnitSquareMesh(2,2, quadrilateral=True)

mass = form.mass(m, 6)
helmholtz = form.helmholtz(m, 6)
hyperelasticity = form.hyperelasticity(m, 6)

parameters = firedrake.parameters['form_compiler'].copy()
parameters['return_impero'] = True




print(complexity(mass, parameters))
print(complexity(helmholtz, parameters))
print(complexity(hyperelasticity, parameters))
