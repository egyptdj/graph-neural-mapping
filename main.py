import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.graphcnn import *
from util import load_data, separate_data
from tqdm import tqdm
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter


c_criterion = nn.CrossEntropyLoss()
d_criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, train_graphs, optimizer, beta, epoch):
    model.train()

    num_rois = train_graphs[0].node_features.shape[1]
    total_iters = args.iters_per_epoch
    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        c_logit, d_logit = model(batch_graph)

        c_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        d_labels = torch.cat([torch.ones(args.batch_size*num_rois, 1), torch.zeros(args.batch_size*num_rois, 1)], 0).to(device)

        d_loss = d_criterion(d_logit, d_labels)
        c_loss = c_criterion(c_logit, c_labels)

        loss = c_loss + beta*d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    average_loss = loss_accum/total_iters
    return average_loss

def pass_data_iteratively(model, graphs):
    model.eval()
    c_logit_list = []
    d_logit_list = []
    for g in graphs:
        c_logit, d_logit = model([g])
        c_logit_list.append(c_logit.detach())
        d_logit_list.append(d_logit)
    return torch.cat(c_logit_list, 0), torch.cat(d_logit_list, 0)


def get_saliency_map(model, graphs, cls):
    model.eval()
    saliency_maps = []
    for graph in graphs:
        saliency_map = model.compute_saliency([graph], cls)
        saliency_maps.append(saliency_map.detach().cpu().numpy())

    saliency_maps = np.stack(saliency_maps, axis=0)
    return saliency_maps


def get_latent_space(model, graphs):
    model.eval()
    output_list = []
    label_list = []
    for g in graphs:
        latent = model([g], latent=True)
        label = np.array([g.label])
        output_list.append(latent)
        label_list.append(label)
    latent_space = np.concatenate(output_list, axis=0)
    labels = np.stack(label_list, axis=0)
    return latent_space, labels


def test(args, model, device, graphs):
    model.eval()
    output, _ = pass_data_iteratively(model, graphs)
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)

    pred = output.max(1, keepdim=True)[1]
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    accuracy = metrics.accuracy_score(labels, pred)
    precision = metrics.precision_score(labels, pred)
    recall = metrics.recall_score(labels, pred)
    return accuracy, precision, recall


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch GIN fMRI')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--sourcedir', type=str, default='data', help='path to the data directory')
    parser.add_argument('--sparsity', type=int, default=30, help='sparsity M of graph adjacency')
    parser.add_argument('--input_feature', type=str, default='one_hot', help='input feature type', choices=['one_hot', 'coordinate', 'mean_bold'])
    parser.add_argument('--batch_size', type=int, default=32, help='input minibatch size for training')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=5, help='learning rate decay step')
    parser.add_argument('--lr_rate', type=float, default=0.8, help='learning rate decay rate')
    parser.add_argument('--fold_seed', type=int, default=0, help='random seed for splitting the dataset')
    parser.add_argument('--fold_idx', type=int, default=0, help='indices of fold in 10-fold validation.')
    parser.add_argument('--num_layers', type=int, default=5, help='number of the GNN layers')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of layers for the MLP. 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units')
    parser.add_argument('--beta', type=float, default=0.05, help='coefficient for infograph regularizer')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"], help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"], help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help='whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--exp', type = str, default = "graph_neural_mapping", help='experiment name')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs, num_classes = load_data(args.sourcedir, args.sparsity, args.input_feature)

    os.makedirs('results/{}/saliency/{}'.format(args.exp, args.fold_idx), exist_ok=True)
    os.makedirs('results/{}/latent/{}'.format(args.exp, args.fold_idx), exist_ok=True)
    os.makedirs('results/{}/model/{}'.format(args.exp, args.fold_idx), exist_ok=True)

    train_graphs, test_graphs = separate_data(graphs, args.fold_seed, args.fold_idx)

    model = GIN_InfoMaxReg(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_rate)

    train_summary_writer = SummaryWriter('results/{}/summary/{}/train'.format(args.exp, args.fold_idx), flush_secs=1, max_queue=1)
    test_summary_writer = SummaryWriter('results/{}/summary/{}/test'.format(args.exp, args.fold_idx), flush_secs=1, max_queue=1)
    with open('results/{}/argv.csv'.format(args.exp), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())

    latent_space_initial, labels = get_latent_space(model, test_graphs)
    np.save('results/{}/latent/{}/latent_space_initial.npy'.format(args.exp, args.fold_idx), latent_space_initial)
    np.save('results/{}/latent/{}/labels.npy'.format(args.exp, args.fold_idx), labels)
    del latent_space_initial
    del labels

    for epoch in tqdm(range(args.epochs), ncols=50, desc=f'{args.fold_idx}'):
        loss_train = train(args, model, device, train_graphs, optimizer, args.beta, epoch)
        scheduler.step()
        acc_train, precision_train, recall_train = test(args, model, device, train_graphs)

        train_summary_writer.add_scalar('loss/total', loss_train, epoch)
        train_summary_writer.add_scalar('metrics/accuracy', acc_train, epoch)
        train_summary_writer.add_scalar('metrics/precision', precision_train, epoch)
        train_summary_writer.add_scalar('metrics/recall', recall_train, epoch)

    acc_test, precision_test, recall_test = test(args, model, device, test_graphs)
    test_summary_writer.add_scalar('metrics/accuracy', acc_test, epoch)
    test_summary_writer.add_scalar('metrics/precision', precision_test, epoch)
    test_summary_writer.add_scalar('metrics/recall', recall_test, epoch)

    torch.save(model.state_dict(), 'results/{}/model/{}/model.pt'.format(args.exp, args.fold_idx))
    latent_space, labels = get_latent_space(model, test_graphs)
    saliency_map_0 = get_saliency_map(model, test_graphs, 0)
    saliency_map_1 = get_saliency_map(model, test_graphs, 1)
    np.save('results/{}/latent/{}/latent_space.npy'.format(args.exp, args.fold_idx), latent_space)
    np.save('results/{}/saliency/{}/saliency_female.npy'.format(args.exp, args.fold_idx), saliency_map_0)
    np.save('results/{}/saliency/{}/saliency_male.npy'.format(args.exp, args.fold_idx), saliency_map_1)


if __name__ == '__main__':
    main()
