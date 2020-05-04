from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Guided_backprop(object):
    """
        Visualize CNN activation maps with guided backprop.

        Returns: An image that represent what the network learnt for recognizing
        the given image.

        Methods: First layer input that minimize the error between the last layers output,
        for the given class, and the true label(=1).

        ! Call visualize(image) to get the image representation
    """
    def __init__(self,model):
        self.model = model
        self.reconstruction = None
        self.activation_maps = []
        self.hooks = []
        # eval mode
        self.predicting_class = []
        self.predicting_class.append(torch.zeros([1,2]).to(self.model.device))
        self.predicting_class.append(torch.zeros([1,2]).to(self.model.device))
        self.predicting_class[0][0,0]=1
        self.predicting_class[1][0,1]=1

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_out, grad_in):
            """ Return reconstructed activation image"""
            self.reconstruction = grad_out[0]

        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_out, grad_in):
            """ Output the grad of model output wrt. layer (only positive) """

            # Gradient of forward_output wrt. forward_input = error of activation map:
                # for relu layer: grad of zero = 0, grad of identity = 1
            grad = self.activation_maps[-1] # corresponding forward pass output
            grad[grad>0] = 1 # grad of relu when > 0

            # set negative output gradient to 0 #!???
            positive_grad_out = torch.clamp(input=grad_out[0],min=0.0)

            # backward grad_out = grad_out * (grad of forward output wrt. forward input)
            new_grad_out = positive_grad_out * grad

            # For hook functions, the returned value will be the new grad_out
            return (new_grad_out,)

        # !!!!!!!!!!!!!!!! change the modules !!!!!!!!!!!!!!!!!!
        # only conv layers, no flattened fc linear layers

        # register hooks to relu layers
        self.hooks.append(self.model.relu.register_forward_hook(forward_hook_fn))
        self.hooks.append(self.model.relu.register_backward_hook(backward_hook_fn))

        # register hook to the first layer
        # first_layer = self.model.mlps[0].linears[0]
        # first_layer.register_backward_hook(first_layer_hook_fn)

    def compute_saliency(self, batch_graph, target_class):
        self.model.eval()
        self.register_hooks()
        model_output, _ = self.model(batch_graph)
        self.model.zero_grad()

        model_output.backward(self.predicting_class[target_class])
        saliency = self.model._last_input.grad.detach().cpu().numpy()
        return saliency

    def release_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image
        import ipdb; ipdb.set_trace()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))
