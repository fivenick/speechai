import sys
import torch
import torch.nn
import torch.nn.functional as F


class TopVirtualNnet(nn.Module):
    '''
    this is a virtual nnet framework at top level and it is applied to the pipline scripts.
    And you should implement four functions after inheriting this object.
    '''
    def __init__(self, *args, **kwargs):
        super(TopVirtualNnet, self).__init__()
        params_dict = locals()
        model_name = str(params_dict["self"]).split("()")[0]
        args_str = utils.iterator_to_params_str(params_dict['args'])
        kwargs_str = utils.dict_to_params_str(params_dict['kwargs'])

        self.model_creation = "{0}({1},{2})".format(model_name, args_str, kwargs_str)

        self.loss = None
        self.use_step = False
        self.transform_keys = []
        self.rename_transform_keys = {}
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        raise NotImplementedError
    def get_model_creation(self):
        return self.model_creation

    @utils.for_device_free
    def forward(self, *inputs):
        raise NotImplementedError

    @utils.for_device_free
    def get_loss(self, *inputs, targets):
        '''
        @return: return a loss tensor, such as return from torch.nn.CrossEntropyLoss(reduction='mean')
        '''
        return self.loss(*inputs, targets)
    def get_posterior(self):
        '''
        @return: return posterior
        '''
        return self.loss.get_posterior()
    @utils.for_device_free
    def get_accuracy(self, targets):
        '''
        @return: return accuracy
        '''
        return self.loss.get_accuracy(targets)
    def auto(self, layer, x):
        '''
        It is convenient for forward-computing when layer could be None or not
        '''
        return layer(x) if layer is not None else x
    def load_transform_state_dict(self, state_dict):
        '''
        It is used in transform-learning.
        '''
        assert isinstance(self.transform_keys, list)
        assert isinstance(self.rename_transform_keys, dict)

        remaining = {utils.key_to_value(self.rename_transform_keys, k, False):v for k,v in state_dict.items() if k.split('.')[0] \
                    in self.transform_keys or k in self.transform_keys }
        self.load_state_dict(remaining, strict=False)
        return self
    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        raise NotImplementedError
    @utils.for_device_free
    def predict(self, outputs):
        '''
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or just softmax for n classes
        @return: an 1-dimensional vector including class-id (0-based) for prediction
        '''
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(outputs, dim=1))
        return prediction
    @utils.for_device_free
    def compute_accuracy(self, outputs, targets):
        '''
        @outpus: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or just softmax for n classes
        @return: the float accuracy
        '''
        assert output.shape[0] == len(targets)
        with torch.no_grad():
            prediction = self.predict(outputs)
            num_correct = (targets==prediction).sum()
        return num_correct.item() / len(targets)
    def step(self, epoch, this_iter, epoch_batchs):
        pass 
    def backward_step(self, epoch, this_iter, epoch_batchs):
        pass 

class ResNetXvector(TopVirtualNnet):
    pass 
