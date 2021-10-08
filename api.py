import torch
from tqdm import trange

class FunFedAlgorithm(object):
    def __init__(self,
                 init_model,
                 make_model,
                 client_dataloaders,
                 distill_dataloder,
                 Dx_loss,
                 loggers,
                 config,
                 device
                 ):
        self.client_dataloaders = client_dataloaders
        self.distill_dataloder = distill_dataloder
        self.make_model = make_model
        self.Dx_loss = Dx_loss
        self.loggers = loggers
        self.config = config
        self.device = device
        self.server_state = self.server_init(init_model)
        self.client_states = [self.client_init(id, self.server_state, client_dataloader) for id, client_dataloader in
                              enumerate(self.client_dataloaders)]

    def step(self, server_state, client_states):
        # server_state contains the (global) model, (global) auxiliary variables, weights of clients
        # client_states contain the (local) auxiliary variables

        # sample active clients
        active_ids = torch.randperm(self.config.n_workers)[:self.config.n_workers_per_round].tolist()

        client_states = self.clients_step(client_states, active_ids)

        # aggregate
        server_state = self.server_step(server_state, client_states, active_ids)

        # broadcast
        client_states = self.clients_update(server_state, client_states, active_ids)

        return server_state, client_states

    def fit(self):
        for round in trange(self.config.n_global_rounds):
            self.server_state, self.client_states = self.step(self.server_state, self.client_states)
            if round % self.config.eval_freq == 0 and self.loggers is not None:
                for logger in self.loggers:
                    logger.log(round, self.server_state.model)

    def server_init(self, init_model):
        raise NotImplementedError

    def client_init(self, id, server_state, client_dataloader):
        raise NotImplementedError

    def clients_step(self, clients_state, active_ids):
        raise NotImplementedError

    def server_step(self, server_state, client_states, active_ids):
        raise NotImplementedError

    def clients_update(self, server_state, clients_state, active_ids):
        raise NotImplementedError





