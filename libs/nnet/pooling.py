import torch
import torch.nn as nn


class StatisticsPooling(nn.Module):
    ''' an usual mean [ + stddev] pooling layer '''
    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(StatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim
        
        self.eps = eps
        self.unbiased = unbiased
    
    def forward(self, inputs):
        '''
        @inputs: a 3-dimensional tensor, including [sample-index, frames-dim-index, frames-index]
        '''
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev:
            if self.unbiased and counts > 1:
                counts = counts - 1
            var = torch.sum((inputs - mean) ** 2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim
    def extra_repr(self):
        return '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'.format(**self.__dict__)
    