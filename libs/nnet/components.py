import torch
import torch.nn as nn

__author__ = 'yxy'

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
        return p

class TdnnAffine(nn.Module):
    '''
    an implemented tdnn affine component by conv1d
    y=splice(w * x, context) + b

    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g. [-2,0,2]
    if context is [0], then the tdnnaffine is equal to linear layer
    '''
    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=True, stride=1, groups=1, norm_w=False, norm_f=False):
        super(TdnnAffine,self).__init__()
        assert input_dim % groups == 0
        # check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if(context[index] >= context[index+1]):
                raise ValueError("context tuple {} is invalid, such as the order.".format(context))
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # it is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0
        self.right_context = context[-1] if context[-1] > 0 else 0

        self.tot_context = self.right_context - self.left_context + 1

        # do not support sphereconv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("warning: do not support sphereconv now and set norm_f=False.")
        
        kernel_size = (self.tot_context,)

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim//groups, *kernel_size))
        # For jit compiling
        # 2021-07-08 Leo
        self.bias = nn.Parameter(torch.randn(output_dim)) if self.bool_bias else None

        self.init_weight()

        # save gpu memory for no skiping case
        if len(context) != self.tot_context:
            self.mask = torch.tensor([[[1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context+1)]]])
        else:
            self.mask = None
        
        self.selected_device = False
    
    def init_weight(self):
        # note, var should be small to avoid slow-shrinking
        nn.init.normal_(self.weight, 0., 0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
    
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        '''
        @inputs: a 3-dimensional tensor, including [samples-index, frames-dim-index, frames-index]
        '''
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode='constant', value=0.0)

        assert inputs.shape[2] >= self.tot_context

        #f for jit compiling
        # 2021-07-08 Leo
        if self.mask is not None:
            self.mask = self.mask.to(inputs.device)
            filters = self.weight * self.mask
        else:
            filters = self.weight

        if self.norm_w:
            filters = F.normalize(filters, dim=1)
        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1,groups=self.groups)
        return outputs
    def extra_reper(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, '\
                'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)


class SEBlock(nn.Module):
    '''
    a se block layer, layer which can learn to use global information to selectively emphasise informative
    features and suppress less useful ones.
    this is a pytorch implementation of se block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
        Snowdar xmuspeech 2020-04-28 [check and update]
    '''
    def __init__(self, input_dim, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the se blocks
        in the network.
        '''
        super(SEBlock, self).__init__()

        self.input_dim = input_dim

        self.fc_1 = TdnnAffine(input_dim, input_dim // ratio)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc_2 = TdnnAffine(input_dim // ratio, input_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        '''
        @inputs: a 3-dimensional tensor, including [smaples-index, frames-dim-index, frames-index]
        '''
        assert len(inputs.shape) == 3

        x = inputs.mean(dim=2, keepdim=True)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))

        return inputs * scale

class SEBlock_2D(nn.Module):
    '''
    a se block layer, layer which can learn to use global information to selectively emphasise informative
    features and suppress less useful ones.
    this is a pytorch implementation of se block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
        Snowdar xmuspeech 2020-04-28 [check and update]
    '''
    def __init__(self, in_planes, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the se blocks
        in the network.
        '''
        super(SEBlock_2D, self).__init__()

        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(in_planes, in_planes // ratio)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc_2 = nn.Linear(in_planes // ratio, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''
        @inputs: a 4-dimensional tensor, including [smaples-index, channel, frames-dim-index, frames-index]
        '''
        b,c,_,_ = inputs.size()
        x = self.avg_pool(inputs).view(b,c)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)

        scale = x.view(b,c,1,1)
        return inputs * scale