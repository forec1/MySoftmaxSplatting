import torch

import sumsplat_cuda as ss


class SumSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenInput, tenFlow):
        ctx.save_for_backward(tenInput, tenFlow)
        return ss.forward(tenInput, tenFlow)

    @staticmethod
    def backward(ctx, gradOutput):
        tenInput, tenFlow = ctx.saved_tensors
        gradInput = ss.backward_input(tenInput, tenFlow, gradOutput)
        gradFlow = ss.backward_flow(tenInput, tenFlow, gradOutput)
        return gradInput, gradFlow


def splatting(x, flow, metric, str_type):
    assert(str_type in ['summation', 'average', 'linear', 'softmax'])
    assert(metric is None or metric.shape[1] == 1)

    if str_type == 'average':
        x = torch.cat([x, x.new_ones(x.shape[0], 1, x.shape[2], x.shape[3])], 1)

    elif str_type == 'linear':
        x = torch.cat([x * metric, metric], 1)

    elif str_type == 'softmax':
        x = torch.cat([x * metric.exp(), metric.exp()], 1)

    output = SumSplatFunction.apply(x, flow)

    if str_type != 'summation':
        output = output[:, :-1, :, :] / (output[:, -1:, :, :] + 0.0000001)

    return output


def softsplat(x, flow, metric):
    x = torch.cat([x * metric.exp(), metric.exp()], 1)
    output = SumSplatFunction.apply(x, flow)
    output = output[:, :-1, :, :] / (output[:, -1:, :, :] + 0.0000001)
    return output


class SoftSplat(torch.nn.Module):
    def forward(self, x, flow, metric):
        return softsplat(x, flow, metric)
