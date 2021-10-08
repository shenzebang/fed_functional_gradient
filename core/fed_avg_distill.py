from api import FunFedAlgorithm
import ray
from collections import namedtuple
from utils.model_utils import FunctionEnsemble
import torch
from torch.optim import SGD
import copy
from core.ffgb_distill import kl_oracle
FEDAVG_D_server_state = namedtuple("FEDAVG_D_server_state", ['global_round', 'model'])
FEDAVG_D_client_state = namedtuple("FEDAVG_D_client_state", ['global_round', 'model'])



class FEDAVG_D(FunFedAlgorithm):
    def __init__(self,
                 init_model,
                 make_model,
                 client_dataloaders,
                 distill_dataloder,
                 Dx_loss,
                 loggers,
                 config,
                 device):
        super(FEDAVG_D, self).__init__(init_model,
                                     make_model,
                                     client_dataloaders,
                                     distill_dataloder,
                                     Dx_loss,
                                     loggers,
                                     config,
                                     device)
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDAVG_D_server_state(global_round=1, model=init_model)

    def client_init(self, server_state, client_dataloader):
        return FEDAVG_D_client_state(global_round=server_state.global_round, model=server_state.model)

    def clients_step(self, clients_state, active_ids):
        print("#" * 30)
        print("start clients step")
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [client_step(self.config, client_state, client_dataloader, self.device)
                                 for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [ray_dispatch.remote(self.config, client_state, client_dataloader, self.device)
                for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state, client_states, active_ids):
        print("#"*30)
        print("start server step")
        active_clients = [client_states[i] for i in active_ids]
        new_model = copy.deepcopy(server_state.model)
        f = FunctionEnsemble()
        for client_state in active_clients:
            f.add_function(client_state.model, 1./len(active_ids))

        new_model = kl_oracle(self.config, f, new_model, self.distill_dataloder, self.device)

        new_server_state = FEDAVG_D_server_state(
            global_round=server_state.global_round + 1,
            model=new_model
        )
        return new_server_state

    def clients_update(self, server_state, clients_state, active_ids):
        return [FEDAVG_D_client_state(global_round=server_state.global_round, model=server_state.model) for _ in clients_state]

@ray.remote(num_gpus=.2)
def ray_dispatch(config, client_state: FEDAVG_D_client_state, client_dataloader, device):
    return client_step(config, client_state, client_dataloader, device)


def client_step(config, client_state: FEDAVG_D_client_state, client_dataloader, device):
    f_local = copy.deepcopy(client_state.model)
    f_local.requires_grad_(True)

    optimizer = SGD(f_local.parameters(), lr=config.fedavg_d_local_lr, weight_decay=config.fedavg_d_weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config.fedavg_d_local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)
            loss.backward()
            optimizer.step()

    return FEDAVG_D_client_state(global_round=client_state.global_round, model=f_local)


