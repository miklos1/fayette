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


@_expression.register(impero.Evaluate)
def _expression_evaluate(expr, temporaries, indices):
    return expression(expr.expression, temporaries, indices, top=True)


@_expression.register(impero.For)
def _expression_for(expr, temporaries, indices):
    return indices[expr.index] * expression(expr.children[0], temporaries, indices)


@_expression.register(impero.Accumulate)
@_expression.register(impero.ReturnAccumulate)
def _expression_accumulate(expr, temporaries, indices):
    return expression(expr.indexsum.children[0], temporaries, indices)


@_expression.register(gem.MathFunction)
def _expression_block(expr, temporaries, indices):
    if expr.name == 'abs':
        return expression(expr.children[0], temporaries, indices) + 1
    
    raise AssertionError("cannot compute complexity polynomial for %s" % type(expr))


class IndexDict(dict):
    """Index dictionary which invents index names as they are required."""
    def __init__(self, *args):
        super().__init__(*args)
        self.count = 0

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            self[key] = sympy.symbols("i%d" % self.count)
            self.count += 1
            return self[key]



def complexity(form, parameters, action=False):
    if action:
        coef = firedrake.Function(form.arguments()[0].function_space())
        form = firedrake.action(form, coef)
    impero_kernel, index_names = tsfc.driver.compile_form(form, parameters=parameters)[0]

    indices = IndexDict({idx: sympy.symbols(name) for idx, name in index_names})

    expr =  expression(impero_kernel.tree, impero_kernel.temporaries, indices, top=True)
    p1 = sympy.symbols("p") + 1
    '''Currently assume p+1 quad points in each direction.'''
    return expr.subs([(i, p1) for i in indices.values()]).expand()


m = ExtrudedMesh(UnitSquareMesh(2, 2, quadrilateral=True), 2)

mass = form.mass(m, 6)
poisson = form.poisson(m, 6)
hyperelasticity = form.hyperelasticity(m, 6)
curl_curl = form.curl_curl(m, 6)

parameters = firedrake.parameters['form_compiler'].copy()
parameters['return_impero'] = True
parameters['mode'] = 'spectral'

for mode, action in (("assembly", False), ("action", True)):
    print(mode)
    print("  mass: ", complexity(mass, parameters, action))
    print("  laplacian: ",complexity(poisson, parameters, action))
    print("  hyperelasticity:", complexity(hyperelasticity, parameters, action))
    print("  curl_curl:", complexity(curl_curl, parameters, action))
