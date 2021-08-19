from torch.autograd import Function


class GradReverse(Function):
    """Gradient reverse layer as implemented in https://github.com/chaoqichen/HTCN."""
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        return grad_output.neg() * ctx.lambd, None
