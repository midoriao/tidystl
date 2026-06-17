# tidystl-compat

Compatibility backends for [tidystl](https://github.com/midoriao/tidystl). Each backend
reproduces, in pure numpy, the robustness semantics of an external STL tool so results can
be compared against that tool without installing it. It registers backends for several
reference tools (Breach, TaLiRo, RTAMT, and more);
call `tidystl.use(tidystl_compat)` and then `tidystl.list_backends()` for the exact set,
e.g. `breach`.

## Install

    pip install tidystl-compat            # numpy backends
    pip install "tidystl-compat[torch]"   # adds the STLCG++ torch backend

## Use

    import tidystl
    import tidystl_compat
    from tidystl import parse, robustness, Signal, list_backends

    tidystl.use(tidystl_compat)

    print(list_backends())  # e.g. ['breach', 'native', 'pymtl', ...]
    rho = robustness(parse("G[0,10](x >= 0)"), sig, backend="breach")
