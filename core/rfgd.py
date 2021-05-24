import torch
from tqdm import tqdm
from utils import FunctionEnsemble, average_function_ensembles, merge_function_ensembles, \
    weak_oracle, get_step_size_scheme

class RFGD:
    def __init__(self, data, label, Dx_loss, n_class, get_init_weak_learner, oracle_steps=1e4,
                 oracle_step_size=0.1, use_residual=True, step_size_0=1., device='cuda', mb_size=128, p=1):
        self.data = data
        self.label = label
        self.n_class = n_class
        self.Dx_loss = Dx_loss
        self.oracle_steps = oracle_steps
        self.oracle_step_size = oracle_step_size
        self.device = device
        self.get_init_weak_learner = get_init_weak_learner
        self.use_residual = use_residual
        f_0 = FunctionEnsemble(get_init_function=get_init_weak_learner, device=device)  # random initialization
        with torch.autograd.no_grad():
            if data.shape[0] <= 5000:
                self.f_data = f_0(data)
            else:  # make sure the the batch size is no more than 5000
                num_chunk = data.shape[0] // 5000 + 1
                self.f_data = torch.cat([f_0(_d) for _d in torch.chunk(data, num_chunk)])
        self.mb_size = mb_size
        self.n_round = 0
        self.step_size_0 = step_size_0
        self.p = p
        if self.use_residual:
            self.residual = torch.zeros(self.data.shape[0], self.n_class, dtype=torch.float32, device=self.device)

    def step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, 1, self.p)
        target = self.Dx_loss(self.f_data, self.label)
        target = target + self.residual if self.use_residual else target
        target = target.detach()
        g, self.residual, g_data = weak_oracle(target, self.data, self.oracle_step_size,
                              self.oracle_steps, init_weak_learner=self.get_init_weak_learner(), mb_size=self.mb_size)
        with torch.autograd.no_grad():
            self.f_data = self.f_data - step_size_scheme(self.n_round) * g_data
        self.n_round += 1

        f_new = FunctionEnsemble(empty=True)
        f_new.add_function(g, -step_size_scheme(self.n_round))
        return f_new
