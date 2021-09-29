from torch.autograd import Function


class GradDampening(Function):
    """Layer with a straight-through forward pass but dampened gradients."""
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        return grad_output * ctx.lambd, None
