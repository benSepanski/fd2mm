from firedrake import assemble, inner, dx, sqrt, grad


def l2_norm_squared(fntn, region=None):
    if region is None:
        return assemble(inner(fntn, fntn) * dx)

    return assemble(inner(fntn, fntn) * dx(region))


def l2_norm(fntn, region=None):
    return sqrt(l2_norm_squared(fntn, region=region))


def h1_norm(fntn, region=None):
    return sqrt(l2_norm_squared(fntn, region=region)
                + l2_norm_squared(grad(fntn), region=region))
