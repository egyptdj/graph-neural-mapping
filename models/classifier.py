import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''
        super(Classifier, self).__init__()

        self.num_layers = num_layers
        self.final_dropout = final_dropout
        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, graph_feature):
        score_over_layer = 0
        for layer, h in enumerate(graph_feature):
            score_over_layer += F.dropout(self.linears_prediction[layer](h), self.final_dropout, training=self.training) # [32,2]

        return score_over_layer
