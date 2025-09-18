import sympy
from sympy.parsing import sympy_parser
from typing import List


def convert_sympy_to_affine_map(
    expr: sympy.core.expr.Expr, symbols: List[sympy.core.symbol.Symbol]
):
    """
    Convert a sympy expression to an MLIR AffineMap.

    Note: this must be called under a module, context and location manager.

    Args:
        expr: sympy expression
        symbols: list of symbols that `expr` might contain

    Returns:
        AffineMap if successful, None if conversion fails
    """

    # Importing lazily to avoid circular import.
    from water_mlir import ir

    def convert_expr(sympy_expr: sympy.core.expr.Expr):
        """Recursively convert sympy expression to AffineExpr"""
        if sympy_expr.is_Integer:
            return ir.AffineExpr.get_constant(sympy_expr.p)

        elif sympy_expr.is_Symbol:
            if sympy_expr in symbols:
                symbol_idx = symbols.index(sympy_expr)
                return ir.AffineExpr.get_symbol(symbol_idx)
            else:
                print(
                    f"Error: Unknown symbol '{sympy_expr}'. Available symbols: {symbols}."
                )
                return None

        elif sympy_expr.is_Rational:
            print(
                f"Warning: Rounding rational number {sympy_expr} down, since affine expr does not support float."
            )
            return ir.AffineExpr.get_floor_div(
                ir.AffineExpr.get_constant(sympy_expr.p),
                ir.AffineExpr.get_constant(sympy_expr.q),
            )

        elif sympy_expr.is_Pow:
            if sympy_expr.exp != -1:
                print(
                    f"Error: Only power of -1 is supported in affine expression, as it can be written as 1/x. Got: {sympy_expr}."
                )
                return None
            # Using floor_div would make the expression be simplified to 0, let's use ceil_div to preserve the division expression.
            print(
                f"Warning: Converting {sympy_expr.base}^(-1) to ceil(1/{sympy_expr.base}) to preserve the division expression."
            )
            return ir.AffineExpr.get_ceil_div(1, convert_expr(sympy_expr.base))

        elif sympy_expr.is_Add:
            result = ir.AffineExpr.get_constant(0)
            for term in sympy_expr.args:
                term_result = convert_expr(term)
                if term_result is None:
                    return None
                result = result + term_result
            return result

        elif sympy_expr.is_Mul:
            result = ir.AffineExpr.get_constant(1)
            divide_by = 1
            for factor in sympy_expr.args:
                # In sympy, x / 2 is expressed as (1/2) * x.
                #           1 / x is expressed as x^(-1).
                # We accumulate the denominator of all these expressions to do a final division at the very end.
                if factor.is_Rational:
                    result *= factor.p
                    divide_by *= factor.q
                    continue
                if factor.is_Pow:
                    assert factor.exp == -1
                    divide_by *= convert_expr(factor.base)
                    continue
                result *= convert_expr(factor)
            if divide_by != 1:
                result = ir.AffineExpr.get_floor_div(result, divide_by)
            return result

        # Handle special functions
        elif hasattr(sympy_expr, "func"):
            func_name = sympy_expr.func.__name__

            if func_name in ("floor", "ceiling"):
                assert len(sympy_expr.args) == 1
                arg = sympy_expr.args[0]
                if arg.is_Mul:
                    # floor(x/2) -> 1/2 = args[0], x = args[1] (Rational args always come first in sympy)
                    if arg.args[0].is_Rational:
                        numerator = convert_expr(arg.args[1]) * arg.args[0].p
                        denominator = arg.args[0].q
                    # floor((x+1)/y) -> 1/y = args[0], x+1 = args[1]
                    elif arg.args[0].is_Pow:
                        numerator = convert_expr(arg.args[1])
                        denominator = convert_expr(arg.args[0].base)
                    # floor(3/x) -> 3 = args[0], 1/x = args[1]
                    elif arg.args[1].is_Pow:
                        numerator = convert_expr(arg.args[0])
                        denominator = convert_expr(arg.args[1].base)
                    else:
                        print(
                            f"Error: Unsupported floor/ceiling expression - {sympy_expr}."
                        )
                        return None
                    return (
                        ir.AffineExpr.get_floor_div(numerator, denominator)
                        if func_name == "floor"
                        else ir.AffineExpr.get_ceil_div(numerator, denominator)
                    )
                if arg.is_Symbol:
                    # Symbols should always be integer, so floor(x) = ceiling(x) = x
                    return convert_expr(arg)
                # Other types of floor()/ceiling() should only involve constants and have been simplified into an integer.
                print(f"Error: Unsupported floor/ceiling expression - {sympy_expr}.")
                return None

            elif func_name == "Mod" and len(sympy_expr.args) == 2:
                x = convert_expr(sympy_expr.args[0])
                y = convert_expr(sympy_expr.args[1])
                return x % y

            else:
                print(
                    f"Error: Unsupported function '{func_name}' in expression: {sympy_expr}."
                )
                return None

        else:
            print(
                f"Error: Unsupported expression type: {type(sympy_expr).__name__} - {sympy_expr}."
            )
            return None

    affine_expr = convert_expr(expr)
    if affine_expr is None:
        return None

    return ir.AffineMap.get(0, len(symbols), [affine_expr])
