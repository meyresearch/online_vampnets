from abc import abstractmethod, ABC
from typing import Dict, Callable, Tuple, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from addict import Dict as Adict
from deeptime.decomposition.deep import vampnet_loss
from tqdm import tqdm_notebook as tqdm 
import numpy as np


class ConfigMixin(ABC):

    DEFAULT = Adict()

    @classmethod
    def get_options(cls, options={}):
        combined_options = Adict(cls.get_default_options())
        combined_options.update(Adict(options))
        # combined_options.version = __version__
        combined_options.runner = cls.__name__
        return combined_options

    @classmethod
    def get_default_options(cls) -> Dict:
        return Adict(cls.DEFAULT)


class VAMPnetEstimator(nn.Module, ConfigMixin):
    DEFAULT = Adict(
        n_hidden_layers=1,
        hidden_layer_width=10,
        output_dim=1,
        input_dim=2,
        output_softmax=False,
        lag_time=1,
        lr=5e-4,
        n_epochs=30,
        optimizer=torch.optim.Adam, 
        score=Adict(
              method='VAMP2', 
              mode='regularize', 
              epsilon=1e-6
        ), 
        loss=vampnet_loss,
        device="cpu"
    )

    def __init__(self, options): 
        super(VAMPnetEstimator, self).__init__()
        self.options = self.get_options(options)

        self.t_0 = self.create_lobe()
        self.t_tau = self.t_0 
        self.optimizer = self.options.optimizer(self.parameters(), lr=self.options.lr)
        if self.options.scheduler is None: 
            self.scheduler = self.options.scheduler(self.optimizer, **self.options.scheduler_kwargs)
        else: 
            self.scheduler = None

        self.device = torch.device(self.options.device)
        self.to(self.device)
        
        self.step = 0
        self.dict_scores = dict({
            "train": {self.options.score.method: {}, "loss": {}},
            "validate": {self.options.score.method: {}, "loss": {}},
            })

    def create_lobe(self):

        dim_inp = self.options.input_dim
        width = self.options.hidden_layer_width
        dim_out = self.options.output_dim
        n_layers = self.options.n_hidden_layers
        lobe = []
        # Input layers
        if n_layers > 1:
            lobe.append(nn.Linear(dim_inp, width))
            lobe.append(nn.ELU())

            for _ in range(n_layers-1):
                lobe.append(nn.Linear(width, width))
                lobe.append(nn.ELU())
        else:
            lobe.append(nn.Linear(dim_inp, width))
            lobe.append(nn.ELU())

        # Output layer
        lobe.append(nn.Linear(width, dim_out))
        if self.options.output_softmax:
            lobe.append(nn.Softmax(dim=1))
        lobe = nn.Sequential(*lobe)
        return lobe

    def forward(self, x):
        x_0 = self.t_0(x[0])
        x_t = self.t_tau(x[1])
        return (x_0, x_t)

    def fit(self, train_loader, validate_loader, record_interval=None, train_callbacks=None, validate_callbacks=None):
        self.optimizer.zero_grad()
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for epoch_ix in range(self.options.n_epochs):
        # for epoch_ix in tqdm(range(self.options.n_epochs), desc='Epoch', total=self.options.n_epochs):
            self.train()
            for batch_ix, batch in enumerate(train_loader):
            # for batch_ix, batch in tqdm(enumerate(train_loader), desc='Batch', total=n_batches):

                self.train_batch(batch, train_callbacks)
                if (batch_ix % record_interval == 0) and (batch_ix > 0):
                    self.eval()
                    if validate_loader is not None: 
                        self.validate(validate_loader, validate_callbacks)
        if self.scheduler is not None: 
            self.scheduler.step()

    def score_batch(self, x):
        x0, xt = x[0].to(self.device), x[1].to(self.device)
        output = self((x0, xt)) # calls the forward method
        loss = self.options.loss(output[0], output[1], **self.options.score)
        return loss

    def train_batch(self,x, callbacks): 
        self.optimizer.zero_grad()
        loss = self.score_batch(x)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        self.dict_scores['train'][self.options.score.method][self.step] = -loss_value
        self.dict_scores['train']['loss'][self.step] = loss_value
        if callbacks is not None:
            for callback in callbacks: 
                callback(self.step, self.dict_scores)
        self.step +=1 

    def validate(self, data_loader, callbacks): 
        losses = []
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
            mean_score = -torch.mean(torch.stack(losses)).item()   

        self.dict_scores['validate'][self.options.score.method][self.step] = mean_score
        self.dict_scores['validate']['loss'][self.step] = -mean_score
        if callbacks is not None:
            for callback in callbacks: 
                callback(self.step, self.dict_scores)


class VAMPNetModel(nn.Module, ConfigMixin):
    DEFAULT = Adict(
        estimator = None, 
        device = 'cpu'
    )
    def __init__(self, options):
        super(VAMPNetModel, self).__init__()
        self.options = self.get_options(options)
        self.device = torch.device(self.options.device)
        self.net = self.options.estimator.t_0
        self.to(self.device)

    def transform(self, data_loader):
        n_batches = len(data_loader)
        self.eval()
        with torch.no_grad():
            out = []
            for batch in tqdm(data_loader, desc='Transform', total=n_batches):
                batch = batch.to(self.device)
                out.append(self.net(batch).detach().cpu().numpy())
        return out


class HedgeVAMPNetEstimator(nn.Module):
    def __init__(self,
                input_dim: int,
                output_dim: int,
                n_hidden_layers: int,
                hidden_layer_width: int,
                loss: Callable[[np.ndarray, np.ndarray], float],
                output_softmax: bool = False,
                device: str = 'cpu',
                b: float = 0.99,
                n: float = 0.01,
                s: float = 0.1
                ):
        super().__init__()

        # Setup device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and 'cuda' in device else "cpu")

        self.loss = loss
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_softmax = output_softmax

        # Setup layers
        hidden_layers = []
        self.n_hidden_layers = n_hidden_layers
        hidden_layers.append(nn.Linear(input_dim, hidden_layer_width))
        for i in range(self.n_hidden_layers-1):
            hidden_layers.append(
                nn.Linear(hidden_layer_width, hidden_layer_width)
            )

        output_layers = []
        for i in range(self.n_hidden_layers):    # connects to hidden layers
            hidden_layer_width = hidden_layers[i].out_features
            output_layers.append(nn.Linear(hidden_layer_width, output_dim))

        self.hidden_t_0 = nn.ModuleList(hidden_layers).to(self.device)
        self.output_t_0 = nn.ModuleList(output_layers).to(self.device)

        # Other training parameters
        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)

        self.alpha = Parameter(torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
                               requires_grad=False).to(
            self.device)

        # Output accumulators
        self.loss_array = []
        self.alpha_array = []

    def partial_forward(self, hidden_module: nn.Module, output_module: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_connections = []
        X = x.to(self.device)
        # X = torch.reshape(X, (self.batch_size, -1))
        # push forward through main network

        hidden_connections.append(F.elu(hidden_module[0](X)))
        for i in range(1, self.n_hidden_layers):
            tmp = hidden_module[i](hidden_connections[i-1])
            hidden_connections.append(F.elu(tmp))

        # push through outputs
        predictions_per_layer = []
        for i in range(self.n_hidden_layers):
            tmp = output_module[i](hidden_connections[i])
            if self.output_softmax:
                predictions_per_layer.append(F.softmax(tmp, dim=1))
            else:
                predictions_per_layer.append(tmp)

        return predictions_per_layer

    def zero_grad(self):
        for i in range(self.n_hidden_layers):
            self.hidden_t_0[i].zero_grad()
            self.output_t_0[i].zero_grad()

    def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        x_0, x_tau = x[0], x[1]
        pred_0_per_layer = self.partial_forward(self.hidden_t_0, self.output_t_0, x_0)
        pred_tau_per_layer = self.partial_forward(self.hidden_t_0, self.output_t_0, x_tau)
        return (pred_0_per_layer, pred_tau_per_layer)

    def loss_per_layer(self, predictions_per_layer: Tuple[List[torch.Tensor]]) -> List[torch.Tensor]:
        losses_per_layer = []

        for pred_0, pred_tau in zip(*predictions_per_layer):
            loss = self.loss(pred_0, pred_tau)
            losses_per_layer.append(loss)
        return losses_per_layer

    def predict(self, x: List[torch.Tensor]) -> float:
        preds_by_layer = self.forward(x)
        loss_by_layer = self.loss_per_layer(preds_by_layer)
        loss_by_layer = torch.stack(loss_by_layer)
        average_loss = torch.sum(torch.mul(self.alpha, loss_by_layer))
        return float(average_loss)

    def transform(self, x: torch.Tensor) -> np.ndarray:
        # Assume stationarity here.
        pred_0_by_layer, _ = self.forward([x, x])
        pred_0_by_layer = torch.stack(pred_0_by_layer)
        # dims are: layers, frames, output states
        a = self.alpha.reshape(self.alpha.shape[0], 1, 1)
        ave_pred_0 = torch.sum(torch.mul(a, pred_0_by_layer), dim=0)
        ave_pred_0 = ave_pred_0.detach().cpu().numpy()
        return ave_pred_0

    def get_alphas(self) -> np.ndarray:
        if self.device.type == 'cuda':
            return self.alpha.to('cpu').numpy()
        else:
            return self.alpha.numpy()


    def update_weights(self, X: List[torch.Tensor]) -> None:
        predictions_per_layer = self.forward(X)
        losses_per_layer = self.loss_per_layer(predictions_per_layer)

        w = [None] * len(losses_per_layer)
        b = [None] * len(losses_per_layer)

        with torch.no_grad():
            for i in range(len(losses_per_layer)):

                losses_per_layer[i].backward(retain_graph=True)
                self.output_t_0[i].weight.data -= self.n * \
                                                   self.alpha[i] * self.output_t_0[i].weight.grad.data
                self.output_t_0[i].bias.data -= self.n * \
                                                 self.alpha[i] * self.output_t_0[i].bias.grad.data

                for j in range(i + 1):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_t_0[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_t_0[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_t_0[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_t_0[j].bias.grad.data

                self.zero_grad()

            for i in range(len(losses_per_layer)):
                self.hidden_t_0[i].weight.data -= self.n * w[i]
                self.hidden_t_0[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / self.n_hidden_layers)

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)

    def partial_fit(self, X: List[torch.Tensor]) -> None:
        self.update_weights(X)


# class HedgeVAMPNetEstimator(nn.Module):
#     def __init__(self, 
#                 input_dim: int, 
#                 output_dim: int, 
#                 n_hidden_layers: int, 
#                 hidden_layer_width: int,
#                 loss: Callable[[np.ndarray, np.ndarray], float],
#                 batch_size: int = 2, 
#                 device: str = 'cpu', 
#                 b: float = 0.99, 
#                 n: float = 0.01, # Weight update learning rate 
#                 s: float = 0.1
#                 ):
#         super().__init__()

#         # Setup device
#         self.device = torch.device(
#             "cuda:0" if torch.cuda.is_available() and 'cuda' in device else "cpu")

#         self.loss = loss
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.batch_size = batch_size
#         assert self.batch_size > 1, "Covariance estimators need two samples at least"
#         # Setup layers
#         hidden_layers = []
#         output_layers = []

#         self.n_hidden_layers = n_hidden_layers + 1  # for batch normalization layer.

#         hidden_layers.append(nn.BatchNorm1d(self.input_dim))  # inputs and outputs are same dimension
#         hidden_layers.append(nn.Linear(input_dim, hidden_layer_width))  # connects 
#         for i in range(self.n_hidden_layers-2): 
#             hidden_layers.append(
#                 nn.Linear(hidden_layer_width, hidden_layer_width)
#             )

#         output_layers.append(nn.Linear(input_dim, output_dim))  # connects to batch normalization layer

#         for i in range(1, self.n_hidden_layers):    # connects to hidden layers
#             output_layers.append(nn.Linear(hidden_layer_width, output_dim)) 

#         self.hidden_t_0 = nn.ModuleList(hidden_layers).to(self.device)
#         self.output_t_0 = nn.ModuleList(output_layers).to(self.device)
        

#         # Other training parameters
#         self.b = Parameter(torch.tensor(
#             b), requires_grad=False).to(self.device)
#         self.n = Parameter(torch.tensor(
#             n), requires_grad=False).to(self.device)
#         self.s = Parameter(torch.tensor(
#             s), requires_grad=False).to(self.device)

#         self.alpha = Parameter(torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
#                                requires_grad=False).to(
#             self.device)

#         # Output accumulators
#         self.loss_array = []
#         self.alpha_array = []

#     def partial_forward(self, hidden_module: nn.Module, output_module: nn.Module, x: torch.Tensor) -> List[torch.Tensor]: 
#         hidden_connections = []
#         X = x.to(self.device)
#         # X = torch.reshape(X, (self.batch_size, -1))
#         # push forward through main network

#         hidden_connections.append(F.elu(hidden_module[0](X)))
#         for i in range(1, self.n_hidden_layers): 
#             tmp = hidden_module[i](hidden_connections[i-1])
#             hidden_connections.append(F.elu(tmp))

#         # push through outputs
#         predictions_per_layer = []
#         for i in range(self.n_hidden_layers): 
#             tmp = output_module[i](hidden_connections[i])
#             predictions_per_layer.append(F.softmax(tmp, dim=1))
#         return predictions_per_layer 

#     def zero_grad(self):
#         for i in range(self.n_hidden_layers):
#             self.hidden_t_0[i].zero_grad()
#             self.output_t_0[i].zero_grad()

#     def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor]]:
#         x_0, x_tau = x[0], x[1]
#         pred_0_per_layer = self.partial_forward(self.hidden_t_0, self.output_t_0, x_0)
#         pred_tau_per_layer = self.partial_forward(self.hidden_t_0, self.output_t_0, x_tau)
#         return (pred_0_per_layer, pred_tau_per_layer)

#     def loss_per_layer(self, predictions_per_layer: Tuple[List[torch.Tensor]]) -> List[torch.Tensor]: 
#         losses_per_layer = []
   
#         for pred_0, pred_tau in zip(*predictions_per_layer):
#             loss = self.loss(pred_0, pred_tau) 
#             losses_per_layer.append(loss)
#         return losses_per_layer


#     def predict(self, x: List[torch.Tensor]) -> float: 
#         preds_by_layer = self.forward(x)
#         loss_by_layer = self.loss_per_layer(preds_by_layer)
#         loss_by_layer = torch.stack(loss_by_layer)
#         average_loss = torch.sum(torch.mul(self.alpha, loss_by_layer))
#         return float(average_loss)

#     def transform(self, x: torch.Tensor) -> np.ndarray: 
#         # Assume stationarity here. 
#         pred_0_by_layer, _ = self.forward([x, x])
#         pred_0_by_layer = torch.stack(pred_0_by_layer)
#         # dims are: layers, frames, output states
#         a = self.alpha.reshape(self.alpha.shape[0], 1, 1)
#         ave_pred_0 = torch.sum(torch.mul(a, pred_0_by_layer), dim=0) 
#         ave_pred_0 = ave_pred_0.detach().cpu().numpy()
#         return ave_pred_0 

#     def get_alphas(self) -> np.ndarray: 
#         if self.device.type == 'cuda': 
#             return self.alpha.to('cpu').numpy()
#         else: 
#             return self.alpha.numpy()


#     def update_weights(self, X: List[torch.Tensor]) -> None: 
#         predictions_per_layer = self.forward(X)
#         losses_per_layer = self.loss_per_layer(predictions_per_layer)

#         w = [None] * len(losses_per_layer)
#         b = [None] * len(losses_per_layer)
        
#         with torch.no_grad():   
#             for i in range(len(losses_per_layer)):

#                 losses_per_layer[i].backward(retain_graph=True)             
#                 self.output_t_0[i].weight.data -= self.n * \
#                                                    self.alpha[i] * self.output_t_0[i].weight.grad.data
#                 self.output_t_0[i].bias.data -= self.n * \
#                                                  self.alpha[i] * self.output_t_0[i].bias.grad.data

#                 for j in range(i + 1):
#                     if w[j] is None:
#                         w[j] = self.alpha[i] * self.hidden_t_0[j].weight.grad.data
#                         b[j] = self.alpha[i] * self.hidden_t_0[j].bias.grad.data
#                     else:
#                         w[j] += self.alpha[i] * self.hidden_t_0[j].weight.grad.data
#                         b[j] += self.alpha[i] * self.hidden_t_0[j].bias.grad.data

#                 self.zero_grad()

#             for i in range(len(losses_per_layer)):
#                 self.hidden_t_0[i].weight.data -= self.n * w[i]
#                 self.hidden_t_0[i].bias.data -= self.n * b[i]

#             for i in range(len(losses_per_layer)):
#                 self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
#                 self.alpha[i] = torch.max(
#                     self.alpha[i], self.s / self.n_hidden_layers)
        
#         z_t = torch.sum(self.alpha)
#         self.alpha = Parameter(
#             self.alpha / z_t, requires_grad=False).to(self.device)

#     def partial_fit(self, X: List[torch.Tensor]) -> None: 
#         self.update_weights(X)