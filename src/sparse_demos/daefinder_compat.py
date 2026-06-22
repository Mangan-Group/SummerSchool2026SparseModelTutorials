"""Compatibility shim for DaeFinder on Python 3.13+.

DaeFinder 0.2.1's ``get_refined_lib`` builds sympy symbols inside a function with
``exec("S, E, ES, P = sympy.symbols(...)")`` and then references those names in a
*separate* ``exec(...)`` call. On Python <= 3.12 the two bare ``exec`` calls shared
the function's locals, so the names persisted. **Python 3.13's PEP 667** ("Consistent
views of namespaces") gives each ``exec`` an independent snapshot of an optimized
frame's locals, so ``exec``-injected names no longer persist across calls -- the
second ``exec`` then raises ``NameError: name 'S' is not defined``.

The fix is to pass an explicit, shared namespace ``dict`` to ``exec`` so the symbols
persist. The shim below is logically identical to the original otherwise, and works
on every supported Python version.

``daeFinder.get_simplified_equation`` (used by notebook 04b to cancel the common
factors a higher-degree library introduces) shares the same PEP 667 issue *and* a
second bug: it mutates ``best_model_df[col].values`` in place, but on modern pandas
that array is a read-only view, raising ``ValueError: assignment destination is
read-only``. The safe reimplementation below fixes both. The real cure for all of
these is an upstream DaeFinder release.
"""
from __future__ import annotations


def get_refined_lib(factor_exp, data_matrix_df_, candidate_library_, get_dropped_feat=False):
    """PEP 667-safe drop-in replacement for ``daeFinder.get_refined_lib``.

    Drops every column of ``candidate_library_`` whose monomial is divisible by any
    expression in ``factor_exp`` (a sympy expr, or a list/set of them).
    """
    import sympy
    from daeFinder.dae_finder import (
        remove_paranth_from_feat, poly_to_scipy, get_factor_feat,
    )

    ns = {"sympy": sympy}                       # <-- shared namespace survives PEP 667
    feat_list = list(data_matrix_df_.columns)
    feat_list_str = ", ".join(remove_paranth_from_feat(data_matrix_df_.columns))
    exec(feat_list_str + " = sympy.symbols(" + str(feat_list) + ")", ns)

    candid_features = remove_paranth_from_feat(poly_to_scipy(candidate_library_.columns))
    candid_feat_dict = {}
    ns["candid_feat_dict"] = candid_feat_dict
    for feat1, feat2 in zip(candidate_library_.columns, candid_features):
        exec("candid_feat_dict['{}'] = {}".format(feat1, feat2), ns)

    dropped_feats = set()
    factors = factor_exp if isinstance(factor_exp, (list, set)) else [factor_exp]
    for factor_ in factors:
        dropped_feats |= set(get_factor_feat(factor_, candid_feat_dict))

    if get_dropped_feat:
        return dropped_feats, candidate_library_.drop(dropped_feats, axis=1)
    return candidate_library_.drop(dropped_feats, axis=1)


def get_simplified_equation(best_model_df, feature, global_feature_list, coef_threshold,
                            intercept_threshold=0.01, intercept=0, simplified=True):
    """PEP 667- and read-only-safe drop-in for ``daeFinder.get_simplified_equation``.

    Builds the discovered relation ``feature = sum(coef * rhs_term)`` as sympy
    expressions and, when ``simplified=True``, cancels any common polynomial factor
    between the two sides (``sympy.cancel(rhs / lhs)``) so a redundant higher-degree
    member of a relation collapses back to its fundamental form. Coefficients below
    ``coef_threshold`` and an intercept below ``intercept_threshold`` are zeroed first.
    """
    import sympy
    from daeFinder.dae_finder import remove_paranth_from_feat, poly_to_scipy

    ns = {"sympy": sympy}                       # shared namespace survives PEP 667
    global_feature_list = list(global_feature_list)
    global_feature_list_string = ", ".join(remove_paranth_from_feat(global_feature_list))
    exec(global_feature_list_string + " = sympy.symbols(" + str(global_feature_list) + ")", ns)

    model_lhs = feature
    model_lhs_sp_string = remove_paranth_from_feat(poly_to_scipy([model_lhs]))[0]

    intercept = 0 if abs(intercept) < intercept_threshold else intercept

    # .values is a read-only view on modern pandas -> take a writable float copy.
    import numpy as np
    model_coefs = np.array(best_model_df[model_lhs].values, dtype=float)
    model_coefs[abs(model_coefs) < coef_threshold] = 0

    model_rhs_features = remove_paranth_from_feat(poly_to_scipy(best_model_df[model_lhs].keys()))

    rhs_terms = [str(coef) + "*" + feat for coef, feat in zip(model_coefs, model_rhs_features)]
    rhs_string_sp_string = "+".join(rhs_terms) + "+" + str(intercept)

    result_dict = {}
    ns["result_dict"] = result_dict
    exec("result_dict['lhs'] = {}".format(model_lhs_sp_string), ns)
    exec("result_dict['rhs'] = {}".format(rhs_string_sp_string), ns)

    if simplified:
        n, d = sympy.fraction(sympy.cancel(result_dict['rhs'] / result_dict['lhs']))
        result_dict['lhs'] = d
        result_dict['rhs'] = n
    return result_dict


def get_simplified_equation_list(best_model_df, global_feature_list, coef_threshold,
                                 intercept_threshold=0.01, intercept_dict=None,
                                 simplified=True, feature_list_=None):
    """PEP 667- and read-only-safe drop-in for ``daeFinder.get_simplified_equation_list``.

    Applies :func:`get_simplified_equation` to each requested LHS feature and returns
    ``{feature: {"lhs": ..., "rhs": ...}}``.
    """
    from copy import deepcopy
    intercept_dict = intercept_dict or {}
    if feature_list_:
        feature_list = deepcopy(feature_list_)
        assert set(feature_list) <= set(best_model_df.columns), \
            "fit for some features missing from the best_model_df"
    else:
        feature_list = best_model_df.columns

    return {feature: get_simplified_equation(best_model_df, feature,
                                             global_feature_list=global_feature_list,
                                             coef_threshold=coef_threshold,
                                             intercept_threshold=intercept_threshold,
                                             intercept=intercept_dict.get(feature, 0),
                                             simplified=simplified)
            for feature in feature_list}


def patch():
    """Install the PEP 667-safe helpers onto the ``daeFinder`` module.

    Patches ``get_refined_lib``, ``get_simplified_equation`` and
    ``get_simplified_equation_list``. Idempotent and harmless on Python <= 3.12.
    Returns the safe ``get_refined_lib`` so a caller can also rebind its own name:
    ``get_refined_lib = daefinder_compat.patch()``.
    """
    import daeFinder
    _safe = {
        "get_refined_lib": get_refined_lib,
        "get_simplified_equation": get_simplified_equation,
        "get_simplified_equation_list": get_simplified_equation_list,
    }
    for name, fn in _safe.items():
        setattr(daeFinder, name, fn)
    try:
        import daeFinder.dae_finder as _mod
        for name, fn in _safe.items():
            setattr(_mod, name, fn)
    except Exception:
        pass
    return get_refined_lib
