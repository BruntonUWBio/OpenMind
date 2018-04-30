import os
import dask.array as da
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
import h5py
from tqdm import tqdm
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.utils import data
from torch.autograd import Variable


class ECoG_NN(nn.Module):
    def __init__(self, in_features, transfer_func):
        super(ECoG_NN, self).__init__()
        self.hidden_list = []
        self.hidden_list.append(nn.Linear(in_features, 1000))
        self.hidden_list.append(nn.Linear(1000, 2000))
        # self.hidden_list.append(nn.Linear(2000, 4000))
        # self.hidden_list.append(nn.Linear(4000, 8000))
        # self.hidden_list.append(nn.Linear(8000, 4000))
        # self.hidden_list.append(nn.Linear(4000, 2000))
        self.hidden_list.append(nn.Linear(2000, 1000))
        self.hidden_list.append(nn.Linear(1000, 50))
        self.hidden_list.append(nn.Linear(50, 1))

        self.hidden_list = nn.ModuleList(self.hidden_list)

        self.transfer_func = transfer_func

    def forward(self, x):
        for layer in self.hidden_list:
            x = self.transfer_func(layer(x))

        return x


def make_all_data(zeros, ones):
    y = [0 for x in zeros]
    y.extend([1 for x in ones])
    all_data = da.concatenate([zeros, ones]).compute()

    return all_data, y


def run_tpot(zeros, ones):
    all_data, y = make_all_data(zeros, ones)
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, y, test_size=.33)
    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    if not os.path.exists('tpot_checkpoint'):
        os.mkdir('tpot_checkpoint')

    if not os.path.exists('tpot_cache'):
        os.mkdir('tpot_cache')
    tpot = TPOTClassifier(
        n_jobs=-1,
        verbosity=3,
        scoring='f1',
        subsample=.5,
        periodic_checkpoint_folder='tpot_checkpoint',
        max_eval_time_mins=20,
        memory='tpot_cache')
    tpot.fit(X_train, y_train)
    tpot.export('tpot_ecog_pipeline.py')
    print(tpot.score(X_test, y_test))


def run_nn(zeros, ones):
    num_epochs = 5000
    all_data, y = make_all_data(zeros, ones)
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, y, test_size=.33)
    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.Tensor(y_train).float()
    y_test = torch.Tensor(y_test).float()
    dataset = data.TensorDataset(X_train.data, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=0)
    model = ECoG_NN(X_train.shape[1], F.sigmoid).cuda()
    learning_rate = 1e-5
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
    sq_losses = []

    for epoch in tqdm(range(num_epochs)):
        for dataset in loader:
            train_y = Variable(dataset[1].float())
            train_x = Variable(dataset[0].float().cuda())
            train_y_hat = model(train_x).cpu()
            model.zero_grad()
            train_loss = F.mse_loss(train_y_hat, train_y).cpu()
            train_loss.backward()
            opt.step()
        # X_train = X_train.to("gpu")
        model = model.cpu()
        all_train_y_hat = model(Variable(X_train))
        # X_train = X_train.to("cpu")
        # X_test = X_test.to("gpu")
        test_y_hat = model(Variable(X_test))
        # X_test = X_test.to("cpu")
        all_train_loss = F.mse_loss(all_train_y_hat, Variable(y_train))
        test_loss = F.mse_loss(test_y_hat, Variable(y_test))
        print(epoch, all_train_loss.data.numpy(), test_loss.data.numpy())
        sq_losses.append([all_train_loss, test_loss])
        model = model.cuda()
    torch.save(model, 'torch_nn')
    train_errs = np.array([x[0].data.numpy() for x in sq_losses]).flatten()
    test_errs = np.array([x[1].data.numpy() for x in sq_losses])
    sns_plot = sns.regplot(range(len(train_errs)), train_errs)
    sns_plot.savefig("train_err")
    sns_plot = sns.regplot(range(len(test_errs)), test_errs)
    sns_plot.savefig("test_err")


def elbow_curve(data):
    data = data[1]
    components = range(1, data.shape[1] + 1)
    explained_variance = []

    for component in tqdm(components[:50]):
        pca = PCA(n_components=component)
        pca.fit(data)
        explained_variance.append(sum(pca.explained_variance_ratio_))
    sns_plot = sns.regplot(
        x=components[:50], y=explained_variance, fit_reg=False).get_figure()
    sns_plot.savefig("pca_elbow.png")


def get_data(data_loc: str) -> tuple:
    data_folders = [os.path.join(data_loc, x) for x in os.listdir(data_loc)]
    out_zeros = None
    out_ones = None

    for data_folder in tqdm(data_folders):
        zero_folder = os.path.join(data_folder, '0')
        one_folder = os.path.join(data_folder, '1')

        if os.path.exists(zero_folder):
            if out_zeros is None:
                out_zeros = da.from_npy_stack(zero_folder)
            else:
                out_zeros = da.concatenate(
                    [out_zeros, da.from_npy_stack(zero_folder)])

        if os.path.exists(one_folder):
            if out_ones is None:
                out_ones = da.from_npy_stack(one_folder)
            else:
                out_ones = da.concatenate(
                    [out_ones, da.from_npy_stack(one_folder)])

    return out_zeros.compute(), out_ones.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ecog_classifier')
    parser.add_argument('-d', required=True, help="Path to data")
    args = vars(parser.parse_args())
    DATA_LOC = args['d']
    zeros, ones = get_data(DATA_LOC)
    run_nn(zeros, ones)
    # elbow_curve(get_data(DATA_LOC))
