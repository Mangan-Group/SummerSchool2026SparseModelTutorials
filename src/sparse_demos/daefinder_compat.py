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
on every supported Python version. (``daeFinder.get_simplified_equation`` has the
same latent issue but is not used by these tutorials; the real cure is an upstream
DaeFinder release.)
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


def patch():
    """Install the PEP 667-safe ``get_refined_lib`` onto the ``daeFinder`` module.

    Idempotent and harmless on Python <= 3.12. Returns the safe function so a caller
    can also rebind its own name: ``get_refined_lib = daefinder_compat.patch()``.
    """
    import daeFinder
    daeFinder.get_refined_lib = get_refined_lib
    try:
        import daeFinder.dae_finder as _mod
        _mod.get_refined_lib = get_refined_lib
    except Exception:
        pass
    return get_refined_lib
