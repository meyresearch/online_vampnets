import torch
import torch.nn as nn
from addict import Dict as Adict
from deeptime.decomposition.deep import vampnet_loss



class VAMPnet(nn.Module):
    DEFAULT = Adict(
        lag_time = 1, 
        network_dimensions = [], 
        lr = 5e-4, 
        n_epochs = 30, 
        optimizer=torch.optim.Adam, 
        score = Adict(
              method='VAMP2', 
              mode='regularize', 
              epsilon=1e-6
        ), 
        loss = vampnet_loss, 
        device="cpu"
    ) 
    def __init__(self, options): 
        super(VAMPnet, self).__init__()
        self.options = self.get_options(options)

        self.t_0 = self.create_lobe()
        self.t_tau = self.t_0 
        self.optimizer = self.options.optimizer(self.parameters(), lr=self.options.lr)
        self.device = torch.device(self.options.device)
        self.to(self.device)
        
        self.step = 0
        self.dict_scores = dict({
            "train": {self.options.score.method: {}, "loss": {}},
            "validate": {self.options.score.method: {}, "loss": {}},
            })

    def create_lobe(self): 
        dims = self.options.network_dimensions
        input_dim = dims[0]
        output_dim = dims[-1]
        lobe = nn.Sequential(nn.BatchNorm1d(input_dim))
        for i in range(1,len(dims)-1):
            in_dim = dims[i-1]
            out_dim = dims[i]
            lobe.append(nn.Linear(in_dim, out_dim))
            lobe.append(nn.ELU()) 
        lobe.append(nn.Linear(out_dim, output_dim))
        lobe.append(nn.Softmax(dim=1))
        return lobe



    @classmethod
    def get_options(cls, options=None):
        if options is None:
            options = {}
        combined_options = Adict(cls.get_default_options())
        combined_options.update(Adict(options))
        # combined_options.version = __version__
        combined_options.feature = cls.__name__
        return combined_options

    @classmethod
    def get_default_options(cls):
        return Adict(cls.DEFAULT)

    def forward(self, x):
        x_0 = self.t_0(x[0])
        x_t = self.t_tau(x[1])
        return (x_0, x_t)


    def fit(self, train_loader, validate_loader): 
        self.optimizer.zero_grad()
        for epoch_ix in range(self.options.n_epochs): 
            self.train()
            for batch in train_loader: 
                self.train_batch(batch)

            self.eval()
            if validate_loader is not None: 
                self.validate(validate_loader)

    def score_batch(self, x):
        x0, xt = x[0].to(self.device), x[1].to(self.device)
        output = self((x0, xt)) # calls the forward method
        loss = self.options.loss(output[0], output[1], **self.options.score)
        return loss

    def train_batch(self,x): 
        self.optimizer.zero_grad()
        loss = self.score_batch(x)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        self.dict_scores['train'][self.options.score.method][self.step] = -loss_value
        self.dict_scores['train']['loss'][self.step] = loss_value
        self.step +=1 

    def validate(self, data_loader): 
        losses = []
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
            mean_score = -torch.mean(torch.stack(losses)).item()   

        self.dict_scores['validate'][self.options.score.method][self.step] = mean_score
        self.dict_scores['validate']['loss'][self.step] = -mean_score

     


