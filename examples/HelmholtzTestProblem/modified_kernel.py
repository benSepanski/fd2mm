import loopy as lp
import numpy as np
from sumpy.kernel import ExpressionKernel, KernelArgument
from sumpy.symbolic import pymbolic_real_norm_2
from pymbolic import var
from pymbolic import make_sym_vector


class ModifiedHelmholtzKernel(ExpressionKernel):
    init_arg_names = ("dim", "helmholtz_k_name", "allow_evanescent")

    def __init__(self, dim=None, helmholtz_k_name="k",
                 allow_evanescent=False):
        """
        :arg helmholtz_k_name: The argument name to use for the Helmholtz
            parameter when generating functions to evaluate this kernel.
        """
        k = var(helmholtz_k_name)

        # Guard against code using the old positional interface.
        assert isinstance(allow_evanescent, bool)

        if dim == 2:
            # H^{(1)}_n'(z) = nH_n^{(1)}(z)/z - H^{(1)}_{n+1}(z)
            # Here, let H_n denote H^{(1)}_n
            #
            #  d/|d|\cdot DH_0(k*r)
            # = d/|d|\cdot d/|d|\cdot -kH_1(k*r)
            # = -kH_1(k*r)
            # so the kernel is
            # K(r) = i/4 * (-kH_1(k*r) - ik*H_0(k*r))
            #      = -ik/4 * (H_1(k*r) + iH_0(k*r))
            # so that
            # K(r) = (d/dr - ik)G(r) where G=i/4*H_0(k*r)
            # is the Helmholtz Green's function
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("hankel_1")(1, k*r) + var("I")*var("hankel_1")(0, k*r)
            scaling = -var("I")*k/4
        elif dim == 3:
            raise UserWarning(
                "ModifiedHelmholtzKernel in dim 3 is just HelmholtzKernel")
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("exp")(var("I")*k*r)/r
            scaling = 1/(4*var("pi"))
        else:
            raise RuntimeError("unsupported dimensionality")

        super(ModifiedHelmholtzKernel, self).__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=True)

        self.helmholtz_k_name = helmholtz_k_name
        self.allow_evanescent = allow_evanescent

    def __getinitargs__(self):
        return (self.dim, self.helmholtz_k_name,
                self.allow_evanescent)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.helmholtz_k_name,
                        self.allow_evanescent))

    def __repr__(self):
        return "ModHelmKnl%dD(%s)" % (
                self.dim, self.helmholtz_k_name)

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
        loopy_knl = lp.register_function_manglers(
            loopy_knl, [bessel_mangler])
        loopy_knl = lp.register_preamble_generators(
            loopy_knl, [bessel_preamble_generator])

        return loopy_knl

    def get_args(self):
        if self.allow_evanescent:
            k_dtype = np.complex128
        else:
            k_dtype = np.float64

        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype),
                    )]

    mapper_method = "map_helmholtz_kernel"


class SplitModifiedHelmholtzKernel(ExpressionKernel):
    init_arg_names = ("dim", "helmholtz_k_name", "allow_evanescent", "nu")

    def __init__(self, dim=None, helmholtz_k_name="k",
                 allow_evanescent=False, nu=None):
        """
        :arg helmholtz_k_name: The argument name to use for the Helmholtz
            parameter when generating functions to evaluate this kernel.
        """
        k = var(helmholtz_k_name)

        # Guard against code using the old positional interface.
        assert isinstance(allow_evanescent, bool)

        if nu is None:
            nu = 0

        if dim == 2:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("hankel_1")(nu, k*r)
            scaling = var("I")/4
        elif dim == 3:
            raise UserWarning(
                "ModifiedHelmholtzKernel in dim 3 is just HelmholtzKernel")
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("exp")(var("I")*k*r)/r
            scaling = 1/(4*var("pi"))
        else:
            raise RuntimeError("unsupported dimensionality")

        super(SplitModifiedHelmholtzKernel, self).__init__(
            dim,
            expression=expr,
            global_scaling_const=scaling,
            is_complex_valued=True)

        self.helmholtz_k_name = helmholtz_k_name
        self.allow_evanescent = allow_evanescent
        self.nu = nu

    def __getinitargs__(self):
        return (self.dim, self.helmholtz_k_name,
                self.allow_evanescent, self.nu)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.helmholtz_k_name,
                                   self.allow_evanescent, self.nu))

    def __repr__(self):
        return "SplitModHelmKnl%dD(%s)(nu%d)" % \
            (self.dim, self.helmholtz_k_name, self.nu)

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
        loopy_knl = lp.register_function_manglers(
            loopy_knl, [bessel_mangler])
        loopy_knl = lp.register_preamble_generators(
            loopy_knl, [bessel_preamble_generator])

        return loopy_knl

    def get_args(self):
        if self.allow_evanescent:
            k_dtype = np.complex128
        else:
            k_dtype = np.float64

        return [KernelArgument(
                loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype),
                )]

    mapper_method = "map_helmholtz_kernel"
