class MLPx1cs(MLPBase):
    def __init__(self, dim_c = None, dim_s = None, dim_x = None,
                 dim_t = None, dim_e = None, actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if dim_t is None: dim_t = discr.dim_t
        if dim_e is None: dim_e = discr.dim_e
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLPx1cs, self).__init__()
        self.dim_c, self.dim_s, self.dim_x, self.dim_t, self.dim_e = dim_c, dim_s, dim_x, dim_t, dim_e
        
        dims_prescs2x = discr.dims_pres2x
        dims_pres2e = discr.dims_pres2e

        # X | C, S
        self.f_vparas2x = mlp_constructor([dim_s + dim_c] + dims_prescs2x + [dim_x], actv, lastactv = False)

        # Env | S
        self.f_vparas2e = mlp_constructor([dim_s] + dims_pres2e + [dim_e], actv, lastactv = False)
 
    def x1cs(self, c, s): return self.f_vparas2x(tc.cat([c, s], dim=-1))
    def e1s(self, s): return self.f_vparas2e(s)
    def forward(self, c, s): return self.x1cs(c, s), self.e1s(s)