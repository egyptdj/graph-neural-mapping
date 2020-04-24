import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import *

from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics

c_criterion = nn.CrossEntropyLoss()
d_criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, train_graphs, optimizer, beta, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    # pbar = tqdm(range(total_iters), unit='batch')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        c_logit, d_logit = model(batch_graph)

        c_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        if args.gcn or args.gcn_cheb or args.gcn_baseline or args.gcn_concat or args.gin_baseline or args.gin_concat:
            d_loss = 0.0
        else:
            d_labels = torch.cat([torch.ones(args.batch_size*int(args.rois.split('_')[-1]), 1), torch.zeros(args.batch_size*int(args.rois.split('_')[-1]), 1)], 0).to(device)
            d_loss = d_criterion(d_logit, d_labels)

        c_loss = c_criterion(c_logit, c_labels)

        #compute loss
        loss = c_loss + beta*d_loss

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()


        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # #report
        # pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters

    return average_loss

# pass data to model without minibatching during testing to avoid memory overflow (does not perform backpropagation)
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
    saliency_maps=[]
    for graph in graphs:
        map = model.compute_saliency([graph], cls)
        saliency_maps.append(map.detach().cpu().numpy())

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
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=5, help='learning rate decay step')
    parser.add_argument('--lr_rate', type=float, default=0.8, help='learning rate decay rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed for splitting the dataset')
    parser.add_argument('--fold_seed', type=int, default=0, help='random seed for splitting the dataset')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation.')
    parser.add_argument('--num_layers', type=int, default=5, help='number of the GNN layers')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of layers for the MLP. 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units')
    parser.add_argument('--beta', type=float, default=0.05, help='coefficient of infograph regularizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='coefficient of l2 weight decay regularizer')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout')
    parser.add_argument('--dropout_layers', nargs='+', default=[], help='layers to apply dropout')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"], help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"], help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help='whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--exp', type = str, default = "graph_neural_mapping", help='experiment name')
    parser.add_argument('--input_feature', type=str, default='one_hot', help='input feature type', choices=['one_hot', 'coordinate', 'mean_bold', 'timeseries_bold'])
    parser.add_argument('--preprocessing', type = str, default = "fixextended", help='HCP run to use', choices=['preproc', 'fixextended'])
    parser.add_argument('--run', type = str, default = "REST1_RL", help='HCP run to use', choices=['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL'])
    parser.add_argument('--rois', type = str, default = "7_400", help='rois [7/17 _ 100/200/300/400/500/600/700/800/900/1000]')
    parser.add_argument('--sparsity', type=int, default=20, help='sparsity K of graph adjacency')
    parser.add_argument('--gcn', action='store_true', help='test the model with gcn')
    parser.add_argument('--gcn_cheb', action='store_true', help='test the model with gcn baseline')
    parser.add_argument('--gcn_baseline', action='store_true', help='test the model with gcn baseline')
    parser.add_argument('--gcn_dgi', action='store_true', help='test the model with gcn baseline')
    parser.add_argument('--gcn_concat', action='store_true', help='test the model with gcn concat')
    parser.add_argument('--gin_baseline', action='store_true', help='test the model with gin baseline')
    parser.add_argument('--gin_concat', action='store_true', help='test the model with gin concat')
    parser.add_argument('--gin_concat_dgi', action='store_true', help='test the model with gin concat')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs('results/{}/saliency/{}'.format(args.exp, args.fold_idx), exist_ok=True)
    os.makedirs('results/{}/latent/{}'.format(args.exp, args.fold_idx), exist_ok=True)
    os.makedirs('results/{}/model/{}'.format(args.exp, args.fold_idx), exist_ok=True)
    os.makedirs('results/{}/csv/{}'.format(args.exp, args.fold_idx), exist_ok=True)

    # graphs, num_classes = load_data(args.preprocessing, args.run, args.rois, args.sparsity, args.input_feature)
    # torch.save(graphs, 'data/graphs.pt')
    graphs = torch.load('data/graphs.pt')
    num_classes = 2

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.fold_seed, args.fold_idx)

    if args.gcn_cheb:
        model = GCN_CAM_Chebconv(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    elif args.gcn:
        model = GCN_InfoMaxReg(5, 1, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, [0,2,3], False, 'average', 'average', device).to(device)
    elif args.gcn_baseline:
        model = GCN_CAM(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    elif args.gcn_concat:
        model = GCN_CAM_concat(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    elif args.gin_baseline:
        model = GIN_CAM(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    elif args.gin_concat:
        model = GIN_CAM_concat(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    elif args.gin_concat_dgi:
        model = GIN_CAM_concat_infomaxreg(train_graphs[0].node_features.shape[1], num_classes, device).to(device)
    else:
        model = GIN_InfoMaxReg(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.dropout_layers, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_rate)

    train_summary_writer = SummaryWriter('results/{}/summary/{}/train'.format(args.exp, args.fold_idx), flush_secs=1, max_queue=1)
    test_summary_writer = SummaryWriter('results/{}/summary/{}/test'.format(args.exp, args.fold_idx), flush_secs=1, max_queue=1)
    with open('results/{}/argv.csv'.format(args.exp), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())

    if not (args.gcn_cheb or args.gcn_baseline or args.gcn_dgi or args.gcn_concat or args.gin_baseline or args.gin_concat or args.gin_concat_dgi):
        latent_space_initial, labels = get_latent_space(model, test_graphs)
        np.save('results/{}/latent/{}/latent_space_initial.npy'.format(args.exp, args.fold_idx), latent_space_initial)
        np.save('results/{}/latent/{}/labels.npy'.format(args.exp, args.fold_idx), labels)
        del latent_space_initial
        del labels

    acc_test_early = 0.0
    precision_test_early = 0.0
    recall_test_early = 0.0
    epoch_early = 0
    for epoch in range(args.epochs):
        loss_train = train(args, model, device, train_graphs, optimizer, args.beta, epoch)
        scheduler.step()
        acc_train,  precision_train, recall_train, = test(args, model, device, train_graphs)

        train_summary_writer.add_scalar('loss/total', loss_train, epoch)
        train_summary_writer.add_scalar('metrics/accuracy', acc_train, epoch)
        train_summary_writer.add_scalar('metrics/precision', precision_train, epoch)
        train_summary_writer.add_scalar('metrics/recall', recall_train, epoch)
        if epoch%25==0: torch.save(model.state_dict(), 'results/{}/model/{}/model.pt'.format(args.exp, args.fold_idx))
        print ('EPOCH [{:3d}] TRAIN_LOSS [{:.3f}] ACC [{:.4f}] P [{:.4f}] R [{:.4f}]'.format(epoch, loss_train, acc_train, precision_train, recall_train))
        acc_test, precision_test, recall_test = test(args, model, device, test_graphs)
        test_summary_writer.add_scalar('metrics/accuracy', acc_test, epoch)
        test_summary_writer.add_scalar('metrics/precision', precision_test, epoch)
        test_summary_writer.add_scalar('metrics/recall', recall_test, epoch)
        with open('results/{}/csv/{}/test_sequence.csv'.format(args.exp, args.fold_idx), 'a') as f:
            f.write(','.join([str(args.fold_idx), str(acc_test), str(precision_test), str(recall_test)]))
            f.write('\n')
        if acc_test > acc_test_early:
            print('top test acc: {:.4f}'.format(acc_test))
            acc_test_early = acc_test
            precision_test_early = precision_test
            recall_test_early = recall_test
            epoch_early = epoch
            torch.save(model.state_dict(), 'results/{}/model/{}/model_early.pt'.format(args.exp, args.fold_idx))
            if not (args.gcn_cheb or args.gcn_baseline or args.gcn_dgi or args.gcn_concat or args.gin_baseline or args.gin_concat or args.gin_concat_dgi):
                latent_space_early, labels = get_latent_space(model, test_graphs)
                saliency_map_0_early = get_saliency_map(model, test_graphs, 0)
                saliency_map_1_early = get_saliency_map(model, test_graphs, 1)
                np.save('results/{}/latent/{}/latent_space_early.npy'.format(args.exp, args.fold_idx), latent_space_early)
                np.save('results/{}/saliency/{}/saliency_female_early.npy'.format(args.exp, args.fold_idx), saliency_map_0_early)
                np.save('results/{}/saliency/{}/saliency_male_early.npy'.format(args.exp, args.fold_idx), saliency_map_1_early)
                del latent_space_early
                del saliency_map_0_early
                del saliency_map_1_early

    print([acc_test, precision_test, recall_test])
    print([acc_test_early, precision_test_early, recall_test_early])

    test_summary_writer.add_scalar('metrics/accuracy', acc_test, epoch)
    test_summary_writer.add_scalar('metrics/precision', precision_test, epoch)
    test_summary_writer.add_scalar('metrics/recall', recall_test, epoch)
    torch.save(model.state_dict(), 'results/{}/model/{}/model.pt'.format(args.exp, args.fold_idx))

    saliency_map_0 = get_saliency_map(model, test_graphs, 0)
    saliency_map_1 = get_saliency_map(model, test_graphs, 1)
    np.save('results/{}/saliency/{}/saliency_female.npy'.format(args.exp, args.fold_idx), saliency_map_0)
    np.save('results/{}/saliency/{}/saliency_male.npy'.format(args.exp, args.fold_idx), saliency_map_1)

    if not (args.gcn_cheb or args.gcn_baseline or args.gcn_dgi or args.gcn_concat or args.gin_baseline or args.gin_concat or args.gin_concat_dgi):
        final_latent_space, _ = get_latent_space(model, test_graphs)
        np.save('results/{}/latent/{}/latent_space_final.npy'.format(args.exp, args.fold_idx), final_latent_space)

    with open('results/{}/csv/{}/result.csv'.format(args.exp, args.fold_idx), 'w') as f:
        f.write(','.join(['fold','epoch', 'accuracy', 'precision', 'recall']))
        f.write("\n")
        f.write(','.join([str(args.fold_idx), str(args.epochs), str(acc_test), str(precision_test), str(recall_test)]))
        f.write("\n")
        f.write(','.join([str(args.fold_idx), str(epoch_early), str(acc_test_early), str(precision_test_early), str(recall_test_early)]))
        f.write("\n")

if __name__ == '__main__':
    main()
