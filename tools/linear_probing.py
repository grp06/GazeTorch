import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from model.head.mlp import MLP
from train.loss_loader import get_loss

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--data_path", type=str)

# debug mode
parser.add_argument("--debug", action='store_true')
parser.set_defaults(debug=False)

# device: "cpu", or an int specifying GPU ID
parser.add_argument("--device", type=str, default='cpu',
help="\"cpu\", or an int specifying GPU ID")

class LinearProbe:
    def __init__(self, feature_dim=512, hidden_dim=256,
        train_loss_type="l1", batch_size=100,
        lr=1e-1,
        device=0,
        debug=False
        ):
        self.gpu = device!='cpu'
        if device=='cpu':
            self.device = torch.device('cpu')
        else:
            try:
                device = int(device)
            except:
                device = 0
            self.device = torch.device(f'cuda:{device}')

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.init_model()

        self.debug = debug

        self.module_loss = get_loss(train_loss_type)
        self.module_ang = get_loss("angular")
        self.batch_size = batch_size

        self.tb = None
        # self.tb = SummaryWriter("temp_results/tb")

    def init_model(self):
        self.model = MLP("main", [[self.feature_dim]], [self.hidden_dim, 2])
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        print("[LP]Model init.")

    def train_epoch(self):
        self.model.train()

        train_ids = np.random.permutation(self.n_train)
        X_train = self.X_train[train_ids, ...]
        Y_train = self.Y_train[train_ids, ...]

        losses = []        
        for i in range(0, self.n_train, self.batch_size):
            if i+self.batch_size > self.n_train: break

            # get a MB
            X = X_train[i:i+self.batch_size, ...]
            Y = Y_train[i:i+self.batch_size, ...]
            if self.gpu:
                X = X.to(self.device)
                Y = Y.to(self.device)

            # FP
            Y_pred = self.model([X])
            loss = self.module_loss(Y, Y_pred)
            losses.append(loss.cpu().detach().numpy())

            # BP
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(losses)

    def val_epoch(self, debug=False):
        self.model.eval()

        losses = []
        results = np.zeros((self.n_test, 2))  
        for i in range(0, self.n_test, self.batch_size):
            if i+self.batch_size > self.n_test: break

            # get a MB
            X = self.X_test[i:i+self.batch_size, ...]
            Y = self.Y_test[i:i+self.batch_size, ...]
            Y_gt = Y.numpy()
            if self.gpu:
                X = X.to(self.device)
                Y = Y.to(self.device)

            # FP
            Y_pred = self.model([X])

            Y_pred_np = Y_pred.cpu().detach().numpy()
            results[i:i+self.batch_size, ...] = Y_pred_np
            if debug and i<1000:
                print(Y_pred_np[0], " P<-***->GT ", Y_gt[0,:])

            loss = self.module_ang(Y, Y_pred)
            losses.append(loss.cpu().detach().numpy())
        return np.mean(losses), results

    def run(self):
        min_loss_eval = 1e6
        min_epoch = 0
        best_results = None
        self.init_model()

        for epoch in tqdm(range(100)):
            loss_train = self.train_epoch()
            if self.tb is not None:
                self.tb.add_scalar("Loss_train", loss_train, global_step=epoch)

            d = self.debug and epoch==99
            loss_val, results = self.val_epoch(debug=d)
            if self.tb is not None:
                self.tb.add_scalar("Loss_val", loss_val, global_step=epoch)

            if loss_val < min_loss_eval:
                min_loss_eval = loss_val
                min_epoch = epoch
                best_results = results
        print(f"Voila. Min val loss = {min_loss_eval} @ {min_epoch}.")
        Y = self.Y_test.numpy()

        loss_p = np.mean(np.abs(Y[:,0] - best_results[:,0]))
        loss_y = np.mean(np.abs(Y[:,1] - best_results[:,1]))
        print("L1 loss(pitch) = ", loss_p)
        print("L1 loss(yaw) = ", loss_y)
        return min_loss_eval, loss_p, loss_y

    def run_mpii(self):
        losses = []
        losses_p = []; losses_y = []
        for fold in range(15):
            print(f"=====FOLD {fold}=====")
            self.mpii_split(fold)
            loss, loss_p, loss_y = self.run()
            losses.append(loss)
            losses_p.append(loss_p)
            losses_y.append(loss_y)
        loss = np.mean(losses)
        loss_p = np.mean(losses_p)
        loss_y = np.mean(losses_y)
        print("=======LOO RESULT=======")
        print("L1 loss = ", loss)
        print("L1 loss(pitch) = ", loss_p)
        print("L1 loss(yaw) = ", loss_y)

    def mpii_init(self, path_X):
        path_Y = "/data/mpiiface/gt_mpiiface_with_conf_v2.npy"
        Y = np.load(path_Y, allow_pickle=True).item()
        self.Y = Y['gaze'].astype(np.float32)

        self.X = np.load(path_X).astype(np.float32)
        # self.X = np.random.randn(45000, 512).astype(np.float32)
        # assume len(X)=len(Y)

    def mpii_split(self, fold=0):
        N_PER_PERSON=3000
        sl1 = slice(0, fold*N_PER_PERSON)
        sl2 = slice(fold*N_PER_PERSON, (fold+1)*N_PER_PERSON)
        sl3 = slice((fold+1)*N_PER_PERSON, None)

        self.X_test = torch.from_numpy(self.X[sl2])
        self.Y_test = torch.from_numpy(self.Y[sl2])
        self.X_train = torch.from_numpy(np.vstack((self.X[sl1], self.X[sl3])))
        self.Y_train = torch.from_numpy(np.vstack((self.Y[sl1], self.Y[sl3])))

        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]

if __name__=='__main__':
    args = parser.parse_args()

    print("LP on: ", args.data_path)
    lp = LinearProbe(device=args.device, debug=args.debug)
    lp.mpii_init(args.data_path)
    lp.run_mpii()