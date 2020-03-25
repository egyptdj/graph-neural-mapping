import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        c_x = c
        c_x_list=[]
        for c in c_x:
            c_x_list.append(c.expand([h_pl.shape[0]//c_x.shape[0],h_pl.shape[1]]))

        c_x = torch.cat(c_x_list, 0)

        sc_1= self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits


#
# class Discriminator(nn.Module):
#     def __init__(self, n_h):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, n_f, g_f, n_idx, s_bias1=None, s_bias2=None): #n_f [557,256], g_f [32,256], n_idx [557]
#         d_criterion = nn.BCEWithLogitsLoss()
#         loss = 0.0
#         logit_list = []
#         for i in range(g_f.shape[0]):
#             logit = self.f_k(n_f, g_f[i,:].repeat(n_f.shape[0],1)).squeeze()
#             logit_list.append(logit)
#             label = (n_idx==i).type_as(logit)
#             loss += d_criterion(logit, label)
#         loss /= g_f.shape[0]
#
#         logits = torch.stack(logit_list)
#
#         return loss, logits
