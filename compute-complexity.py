import form
import firedrake
import tsfc
from gem import gem
from gem import impero
from functools import reduce, singledispatch
import sympy


from firedrake import COMM_WORLD, ExtrudedMesh, UnitCubeMesh, UnitSquareMesh, assemble

m = UnitSquareMesh(2,2, quadrilateral=True)

mass = form.mass(m, 6)

parameters = firedrake.parameters['form_compiler'].copy()
parameters['return_impero'] = True

impero_kernel, index_names = tsfc.driver.compile_form(mass, parameters=parameters)[0]
c_kernel = tsfc.driver.compile_form(mass, parameters=firedrake.parameters['form_compiler'])[0]

indices={idx: sympy.symbols(name) for idx, name in index_names}

def expression(expr, ):
    """Translates GEM expression into a COFFEE snippet, stopping at
    temporaries.

    :arg expr: GEM expression
    :arg parameters: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: COFFEE expression
    """
    if not top and expr in parameters.names:
        return _ref_symbol(expr, parameters)
    else:
        return _expression(expr, parameters)


def expression(expr, temporaries, top=False):
    """Walks an impero tree and computes the complexity polynomial

    :returns: Sympy polynomial.
    """
    if not top and expr in temporaries:
        return 0
    else:
        return _expression(expr, temporaries)


@singledispatch
def _expression(expr, temporaries):
    raise AssertionError("cannot compute complexity polynomial for %s" % type(expr))


@_expression.register(gem.Product)
def _expression_product(expr, temporaries):
    return expression(expr.children[0], temporaries) + expression(expr.children[1], temporaries) + 1


@_expression.register(gem.Sum)
def _expression_sum(expr, temporaries):
    return expression(expr.children[0], temporaries) + expression(expr.children[1], temporaries) + 1


@_expression.register(gem.Division)
def _expression_division(expr, temporaries):
    return expression(expr.children[0], temporaries) + expression(expr.children[1], temporaries) + 1


@_expression.register(gem.Constant)
def _expression_scalar(expr, temporaries):
    return 0

@_expression.register(gem.Variable)
def _expression_variable(expr, temporaries):
    return 0


@_expression.register(gem.Indexed)
def _expression_indexed(expr, temporaries):
    return expression(expr.children[0], temporaries)


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, temporaries):
    return expression(expr.children[0], temporaries)


@_expression.register(impero.Block)
def _expression_block(expr, temporaries):
    return reduce(lambda x, y: x+y, map(lambda e: expression(e, temporaries), expr.children))


@_expression.register(impero.Evaluate)
def _expression_evaluate(expr, temporaries):
    return _expression(expr.expression, temporaries)


@_expression.register(impero.For)
def _expression_for(expr, temporaries):
    return indices[expr.index] * expression(expr.children[0], temporaries)


@_expression.register(impero.Initialise)
def _expression_initialise(expr, temporaries):
    return expression(expr.indexsum, temporaries)


@_expression.register(impero.Accumulate)
def _expression_initialise(expr, temporaries):
    return _expression(expr.indexsum, temporaries)

@_expression.register(impero.ReturnAccumulate)
def _expression_initialise(expr, temporaries):
    return _expression(expr.indexsum, temporaries)


@_expression.register(gem.IndexSum)
def _expression_block(expr, temporaries):
    return reduce(lambda x, y: x*y, map(lambda i: indices[i], expr.multiindex)) * expression(expr.children[0], temporaries) 


@_expression.register(gem.MathFunction)
def _expression_block(expr, temporaries):
    if expr.name == 'abs':
        return expression(expr.children[0], temporaries) + 1
    
    raise AssertionError("cannot compute complexity polynomial for %s" % type(expr))



complexity = expression(impero_kernel.tree, impero_kernel.temporaries, top=True)
