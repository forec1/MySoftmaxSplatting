import torch

import matplotlib.pyplot as plt
import numpy as np

import softsplat


def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(np.frombuffer(strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

    intWidth = np.frombuffer(strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(strFlow, dtype=np.int32, count=1, offset=8)[0]

    return np.frombuffer(strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape([intHeight, intWidth, 2])


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).cuda()

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)


imgFirst = plt.imread('../../resources/first.png')
imgSecond = plt.imread('../../resources/second.png')
flow = read_flo('../../resources/flow.flo')

tenFirst = torch.FloatTensor(imgFirst.transpose(2, 0, 1)[None, :, :, :]).cuda()
tenSecond = torch.FloatTensor(imgSecond.transpose(2, 0, 1)[None, :, :, :]).cuda()
tenFlow = torch.FloatTensor(flow.transpose(2, 0, 1)[None, :, :, :]).cuda()

tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
    tenSoftmax = softsplat.splatting(tenFirst, tenFlow * fltTime, -20.0 * tenMetric, 'softmax')
    tenSummation = softsplat.splatting(tenFirst, tenFlow * fltTime, None, 'summation')
    tenAverage = softsplat.splatting(tenFirst, tenFlow * fltTime, None, 'average')
    tenLinear = softsplat.splatting(tenFirst, tenFlow * fltTime, (0.3 - tenMetric).clamp(0.0000001, 1), 'linear')

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(tenSoftmax[0,:, :, :].cpu().numpy().transpose(1, 2, 0))
    axs[0, 0].title.set_text('Softmax splatting')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(tenSummation[0,:, :, :].cpu().numpy().transpose(1, 2, 0))
    axs[1, 0].title.set_text('Summation splatting')
    axs[1, 0].axis('off')

    axs[0, 1].imshow(tenAverage[0,:, :, :].cpu().numpy().transpose(1, 2, 0))
    axs[0, 1].title.set_text('Average splatting')
    axs[0, 1].axis('off')

    axs[1, 1].imshow(tenLinear[0,:, :, :].cpu().numpy().transpose(1, 2, 0))
    axs[1, 1].title.set_text('Linear splatting')
    axs[1, 1].axis('off')
    plt.show()
