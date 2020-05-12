from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
from softsplat import SumSplatFunction

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
parser.add_argument('-s', '--save', type=str, default=None)
options = parser.parse_args()

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size, 3, options.features, options.features, **kwargs)
F = torch.randn(options.batch_size, 2, options.features, options.features, **kwargs)
sumsplat = SumSplatFunction.apply

# Force CUDA initialization
output = sumsplat(X, F)

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    start = time.time()
    output = sumsplat(X, F)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    (output.sum()).backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
forward_average = forward_time / options.runs * scale

backward_min *= scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.6f}/{1:.6f} {4} | Backward {2:.6f}/{3:.6f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))
if options.save:
    with open(options.save, "w") as f:
        f.write(str(forward_min) + ", " + str(forward_average) + "\n")
        f.write(str(backward_min) + ", " + str(backward_average))
