import torch

import sumsplat_cuda as ss


class SoftSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenInput, tenFlow, importance_mask):
        ctx.save_for_backward(tenInput, tenFlow)
        tenInput = torch.cat([tenInput * importance_mask.exp(), importance_mask.exp()], 1)
        tenOutput = ss.forward(tenInput, tenFlow)
        tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001)
        return tenOutput

    @staticmethod
    def backward(ctx, gradOutput):
        tenInput, tenFlow = ctx.saved_tensors
        gradInput = ss.backward_input(tenInput, tenFlow, gradOutput)
        gradFlow = ss.backward_flow(tenInput, tenFlow, gradOutput)
        return gradInput, gradFlow


class SumSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenInput, tenFlow):
        return ss.forward(tenInput, tenFlow)


class AvgSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenInput, tenFlow):
        tenInput = torch.cat([tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])], 1)
        tenOutput = ss.forward(tenInput, tenFlow)
        tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001)
        return tenOutput


class LinearSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenInput, tenFlow, importance_mask):
        tenInput = torch.cat([tenInput * importance_mask, importance_mask], 1)
        tenOutput = ss.forward(tenInput, tenFlow)
        tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001)
        return tenOutput


class SoftSplat(torch.nn.Module):
    def forward(self, tenInput, tenFlow, importance_mask):
        return SoftSplatFunction.apply(tenInput, tenFlow, importance_mask)
