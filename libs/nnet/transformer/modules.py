import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'yxy'

class ScaledDotProductAttention(nn.Module):
    ''' scaled dot-product attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2,3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class DFsmn(nn.Module):
    ''' alibaba dfsmn module '''
    def __init__(self, l_order, r_order, l_stride, r_stride, d_v):
        super().__init__()
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride
        l_filter = torch.randn(l_order, d_v, requires_grad=True)
        r_filter = torch.randn(r_order, d_v, requires_grad=True)
        self.l_filter = nn.Parameter(l_filter)
        self.r_filter = nn.Parameter(r_filter)
    '''
    def forward(self, v):
        p = v.clone()
        v_shape = v.shape

        for batch in range(v_shape[0]):
            for head in range(v_shape[1]):
                for frame in range(v_shape[2]):
                    order = 0
                    orders = []
                    shift_indexs = []
                    while order < self.l_order:
                        shift_index = frame - order * self.l_stride
                        if shift_index < 0:
                            break
                        orders.append(order)
                        shift_indexs.append(shift_index)
                        order += 1

                    p[batch, head, frame] += torch.sum(self.l_filter[orders]*p[batch, head, shift_indexs], dim=0)

                    order = 1
                    orders = []
                    shift_indexs = []
                    while order < (self.r_order+1):
                        shift_index = frame + order * self.r_stride
                        if shift_index >= v_shape[2]:
                            break
                        orders.append(order - 1)
                        shift_indexs.append(shift_index)
                        order += 1

                    p[batch, head, frame] += torch.sum(self.r_filter[orders]*p[batch, head, shift_indexs], dim=0)

                    #del temp1
                    #del temp
                    #torch.cuda.empty_cache()
        return p
    '''

    def forward(self, v):
        p = v.clone().detach()
        v_shape = v.shape

        for frame in range(v_shape[2]):
            order = 0
            orders = []
            shift_indexs = []
            while order < self.l_order:
                shift_index = frame - order * self.l_stride
                if shift_index < 0:
                    break
                orders.append(order)
                shift_indexs.append(shift_index)
                order += 1

            p[:, :, frame] += torch.sum(self.l_filter[orders] * p[:, :, shift_indexs], dim=2)

            order = 1
            orders = []
            shift_indexs = []
            while order < (self.r_order+1):
                shift_index = frame + order * self.r_stride
                if shift_index >= v_shape[2]:
                    break
                orders.append(order - 1)
                shift_indexs.append(shift_index)
                order += 1

            p[:, :, frame] += torch.sum(self.r_filter[orders] * p[:, :, shift_indexs], dim=2)

            #del temp1
            #del temp
            #torch.cuda.empty_cache()
        return p
