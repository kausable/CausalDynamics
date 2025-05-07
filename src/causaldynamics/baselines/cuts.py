import warnings

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor, nn


class CUTSPlus:
    """
    CUTSPlus baseline.

    Reference:
        [1] https://github.com/jarrycyx/UNN/blob/main/CUTS_Plus/cuts_plus_example.ipynb
    """

    def __init__(self, tau_max: int = 1, corr_thres: float = 0.8):
        """Initialize regressor"""
        super(CUTSPlus, self).__init__()
        self.tau_max = tau_max
        self.corr_thres = corr_thres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, X, verbosity: int = 0):
        """Estimate lagged adjacency graph"""
        self.adj_matrix = main(
            X,
            mask=np.ones_like(X),
            true_cm=None,
            opt=default_opt.sota.cuts_plus,
            device=self.device,
        )

        self.adj_matrix = (self.adj_matrix - self.adj_matrix.min()) / (
            self.adj_matrix.max() - self.adj_matrix.min()
        )
        self.adj_matrix = self.adj_matrix > self.corr_thres


##########################################################################################################################
default_opt = OmegaConf.create(
    {
        "log": {"stdout": False, "stderr": False, "tensorboard": False},
        "sota": {
            "cuts_plus": {
                "n_nodes": "auto",  # will be set to data.shape[-1] if auto
                "input_step": 1,
                "batch_size": 512,
                "data_dim": 1,
                "total_epoch": 10,
                "n_groups": 1,
                "group_policy": "multiply_1_every_20",
                "supervision_policy": "full",
                "fill_policy": "rate_0.2_after_20",
                "show_graph_every": 64,
                "data_pred": {
                    "model": "multi_lstm",
                    "pred_step": 1,
                    "mlp_hid": 16,
                    "gru_layers": 1,
                    "shared_weights_decoder": False,
                    "concat_h": True,
                    "lr_data_start": 1e-2,
                    "lr_data_end": 1e-3,
                    "weight_decay": 0,
                    "prob": True,
                },
                "graph_discov": {
                    "lambda_s_start": 1e-1,
                    "lambda_s_end": 1e-2,
                    "lr_graph_start": 1e-3,
                    "lr_graph_end": 1e-4,
                    "start_tau": 1,
                    "end_tau": 0.1,
                    "dynamic_sampling_milestones": [0],
                    "dynamic_sampling_periods": [1],
                },
            }
        },
    }
)


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> Tensor:
    """
    Samples from the Gumbel-Softmax distribution and optionally discretizes.

    Args:
        logits: Tensor with unnormalized log probabilities.
        tau: Temperature parameter.
        hard: If True, returns one-hot vectors via a straight-through estimator.
        eps: Deprecated epsilon.
        dim: Dimension for softmax.

    Returns:
        A tensor of the same shape as logits.
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")
    # Sample Gumbel noise.
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    # Add noise and scale by temperature.
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        # Get one-hot vectors.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class GRUCell(nn.Module):
    """
    Custom GRU cell using message passing via MPNN.
    """

    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation="tanh"):
        super(GRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)
        mpnn_channel = d_in * n_nodes + num_units if concat_h else d_in * n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)
        c = self.activation_fn(c)
        return u * h + (1 - u) * c


class MPNN(nn.Module):
    """
    Message Passing Neural Network module using Conv1d.
    """

    def __init__(self, c_in, c_out, concat_h=True):
        super(MPNN, self).__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x, h, graph):
        b, c, n = x.shape
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n)
        x_messages = torch.einsum("bcmn,bmn->bcmn", (x_repeat, graph))
        x_messages = rearrange(x_messages, "b c m n -> b (c m) n")
        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        return out


class LocalConv1D(nn.Module):
    """
    Local 1D convolution applied to each node separately.
    """

    def __init__(self, in_channels, out_channels, kernel_size, n_nodes):
        super(LocalConv1D, self).__init__()
        self.out_channel = out_channels
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
                for _ in range(n_nodes)
            ]
        )

    def forward(self, x):
        b, h, n = x.shape
        out = torch.zeros((b, self.out_channel, n), device=x.device)
        for i in range(n):
            x_local_in = x[..., i].unsqueeze(-1)
            x_local_out = self.conv_list[i](x_local_in)
            out[..., i] = x_local_out.squeeze(-1)
        return out


class CUTS_Plus_Net(nn.Module):
    """
    Neural network for latent data prediction and graph discovery.
    Combines a convolutional encoder, GRU cells, and a decoder.
    """

    def __init__(
        self,
        n_nodes,
        in_ch=1,
        hidden_ch=32,
        n_layers=1,
        shared_weights_decoder=False,
        concat_h=False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        self.conv_encoder1 = nn.Conv1d(
            in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1
        )
        self.conv_encoder2 = nn.Conv1d(
            in_channels=2 * hidden_ch, out_channels=hidden_ch, kernel_size=1
        )
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=2 * hidden_ch, out_channels=in_ch, kernel_size=1)
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(
                    in_channels=2 * hidden_ch,
                    out_channels=in_ch,
                    kernel_size=1,
                    n_nodes=n_nodes,
                )
            )
        self.act = nn.LeakyReLU()
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(
                GRUCell(
                    d_in=in_ch if i == 0 else hidden_ch,
                    num_units=hidden_ch,
                    n_nodes=n_nodes,
                    concat_h=concat_h,
                )
            )
        self.h0 = self.init_state(n_nodes)

    def init_state(self, n_nodes):
        h = []
        for _ in range(self.n_layers):
            h.append(nn.parameter.Parameter(torch.zeros([self.hidden_ch, n_nodes])))
        return nn.ParameterList(h)

    def update_state(self, x, h, graph):
        rnn_in = x
        for layer in range(self.n_layers):
            rnn_in = h[layer] = self.cells[layer](rnn_in, h[layer], graph)
        return h

    def forward(self, x, mask, fwd_graph):
        x = rearrange(x, "b n s c -> b c n s")
        bs, in_ch, n_nodes, steps = x.shape
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        pred = []
        for step in range(steps):
            x_now = x[..., step]
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]
            x_repr = self.act(self.conv_encoder1(h_now))
            x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1)))
            x_repr = torch.cat([x_repr, h_now], dim=1)
            x_hat2 = self.decoder(x_repr)
            pred.append(x_hat2)
        pred = torch.stack(pred, dim=-1)
        pred = rearrange(pred, "b c n s -> b n s c")
        return pred[:, :, -1:]


def generate_indices(input_step, pred_step, t_length, block_size=None):
    if block_size is None:
        block_size = t_length
    offsets_in_block = np.arange(input_step, block_size - pred_step + 1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()
    np.random.shuffle(random_t_list)
    return random_t_list


def batch_generater(
    data, observ_mask, bs, n_nodes, input_step, pred_step, block_size=None
):
    t_length, n, d = data.shape
    random_t_list = generate_indices(input_step, pred_step, t_length, block_size)
    for _ in range(len(random_t_list) // bs):
        x = torch.zeros([bs, n_nodes, input_step, d]).to(data.device)
        y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        t_idx = torch.zeros([bs]).to(data.device).long()
        mask_x = torch.zeros([bs, n_nodes, input_step, d]).to(data.device)
        mask_y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
        for data_i in range(bs):
            curr_t = random_t_list.pop()
            x[data_i] = rearrange(
                data[curr_t - input_step : curr_t, :], "t n d -> n t d"
            )
            y[data_i] = rearrange(
                data[curr_t : curr_t + pred_step, :], "t n d -> n t d"
            )
            t_idx[data_i] = curr_t
            mask_x[data_i] = rearrange(
                observ_mask[curr_t - input_step : curr_t, :], "t n d -> n t d"
            )
            mask_y[data_i] = rearrange(
                observ_mask[curr_t : curr_t + pred_step, :], "t n d -> n t d"
            )
        yield x, y, t_idx, mask_x, mask_y


class MultiCAD(object):
    """
    Main class for causal graph discovery and latent data prediction.
    """

    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device

        # Initialize the data prediction network.
        self.fitting_model = CUTS_Plus_Net(
            self.args.n_nodes,
            in_ch=self.args.data_dim,
            n_layers=self.args.data_pred.gru_layers,
            hidden_ch=self.args.data_pred.mlp_hid,
            shared_weights_decoder=self.args.data_pred.shared_weights_decoder,
            concat_h=self.args.data_pred.concat_h,
        ).to(self.device)

        # Loss and optimizer for data prediction.
        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(
            self.fitting_model.parameters(),
            lr=self.args.data_pred.lr_data_start,
            weight_decay=self.args.data_pred.weight_decay,
        )

        # Set up learning rate scheduler.
        if "every" in self.args.fill_policy:
            lr_schedule_length = int(self.args.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.args.total_epoch
        gamma = (
            self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start
        ) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma
        )

        self.n_groups = self.args.n_groups
        if self.args.group_policy == "None":
            self.args.group_policy = None

        # Initialize Gumbel and sparsity parameters for graph discovery.
        end_tau, start_tau = (
            self.args.graph_discov.end_tau,
            self.args.graph_discov.start_tau,
        )
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        end_lmd, start_lmd = (
            self.args.graph_discov.lambda_s_end,
            self.args.graph_discov.lambda_s_start,
        )
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd

        # IMPORTANT: Create learnable GT as a leaf tensor on the correct device.
        self.GT = nn.Parameter(
            torch.ones((self.n_groups, self.args.n_nodes), device=self.device) * 0.5,
            requires_grad=True,
        )
        if self.n_groups == self.args.n_nodes:
            self.G = torch.eye(self.args.n_nodes, device=self.device)
        else:
            self.G = torch.zeros((self.args.n_nodes, self.n_groups), device=self.device)
        self.set_graph_optimizer(0)

    def set_graph_optimizer(self, epoch=None):
        if epoch is None:
            epoch = 0
        gamma = (
            self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start
        ) ** (1 / self.args.total_epoch)
        self.graph_optimizer = torch.optim.Adam(
            [self.GT], lr=self.args.graph_discov.lr_graph_start * gamma**epoch
        )
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(
            self.graph_optimizer, step_size=1, gamma=gamma
        )

    def latent_data_pred(self, x, y, mask_x, mask_y):
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()

        bs, n, t, d = x.shape
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()
        GT_prob = self.GT
        G_prob = self.G
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = sample_bernoulli(Graph, self.args.batch_size)
        y_pred = self.fitting_model(x, mask_x, graph_sampled)
        loss = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y, mask_x, mask_y):
        def gumbel_sigmoid_sample(graph, batch_size, tau=1):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples

        _, n = self.GT.shape
        self.graph_optimizer.zero_grad()
        GT_prob = self.GT
        G_prob = self.G
        Graph = torch.einsum("nm,ml->nl", G_prob, torch.sigmoid(GT_prob))
        graph_sampled = gumbel_sigmoid_sample(Graph, self.args.batch_size)
        loss_sparsity = torch.linalg.norm(Graph.flatten(), ord=1) / (n * n)
        y_pred = self.fitting_model(x, mask_x, graph_sampled)
        loss_data = self.data_pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(
            mask_y
        )
        loss = loss_sparsity * self.lambda_s + loss_data
        loss.backward()
        self.graph_optimizer.step()
        return loss, loss_sparsity, loss_data

    def train(self, data, observ_mask, original_data, true_cm=None):
        original_data = torch.from_numpy(original_data).float().to(self.device)
        observ_mask = torch.from_numpy(observ_mask).float().to(self.device)
        data = torch.from_numpy(data).float().to(self.device)
        if self.args.supervision_policy == "full":
            observ_mask = torch.ones_like(observ_mask)
        latent_pred_step = 0
        graph_discov_step = 0
        data_interp = data.clone()
        original_mask = observ_mask.clone()
        for epoch_i in range(self.args.total_epoch):
            if self.args.group_policy is not None:
                group_mul = int(self.args.group_policy.split("_")[1])
                group_every = int(self.args.group_policy.split("_")[3])
                if epoch_i % group_every == 0 and self.n_groups < self.args.n_nodes:
                    if epoch_i != 0:
                        self.n_groups *= group_mul
                    if self.n_groups > self.args.n_nodes:
                        self.n_groups = self.args.n_nodes
                    self.G = torch.zeros(
                        [self.args.n_nodes, self.n_groups], device=self.device
                    )
                    for i in range(0, self.n_groups):
                        for j in range(0, self.args.n_nodes // self.n_groups):
                            self.G[i * (self.args.n_nodes // self.n_groups) + j, i] = 1
                    for k in range(
                        i * (self.args.n_nodes // self.n_groups) + j, self.args.n_nodes
                    ):
                        self.G[k, i] = 1
                    if hasattr(self, "GT"):
                        GT_init = (
                            torch.sigmoid(self.GT)
                            .detach()
                            .cpu()
                            .repeat_interleave(group_mul, 0)[: self.n_groups, :]
                        )
                        GT_init = 1 - (1 - GT_init) ** (1 / group_mul)
                    else:
                        GT_init = torch.ones((self.n_groups, self.args.n_nodes)) * 0.5
                    self.GT = nn.Parameter(GT_init.to(self.device))
                    self.set_graph_optimizer(epoch_i)
                elif epoch_i == 0 and self.n_groups == self.args.n_nodes:
                    self.G = torch.eye(self.args.n_nodes, device=self.device)
                    GT_init = (
                        torch.ones(
                            (self.n_groups, self.args.n_nodes), device=self.device
                        )
                        * 0.5
                    )
                    self.GT = nn.Parameter(GT_init)
                    self.set_graph_optimizer(epoch_i)
            if "every" in self.args.fill_policy:
                update_every = int(self.args.fill_policy.split("_")[-1])
                if (epoch_i + 1) % update_every == 0:
                    data = data_pred
                    self.data_pred_optimizer.param_groups[0][
                        "lr"
                    ] = self.args.data_pred.lr_data_start
                    observ_mask = torch.ones_like(original_mask)
            elif "rate" in self.args.fill_policy:
                update_rate = float(self.args.fill_policy.split("_")[1])
                update_after = int(self.args.fill_policy.split("_")[3])
                if epoch_i + 1 > update_after:
                    data = data * (1 - update_rate) + data_pred * update_rate
            if "masked_before" in self.args.supervision_policy:
                masked_before = int(self.args.supervision_policy.split("_")[2])
                if epoch_i == masked_before:
                    observ_mask = torch.ones_like(original_mask)
                    self.gumbel_tau = self.start_tau
            if hasattr(self.args, "data_pred"):
                block_size = (
                    self.args.block_size if hasattr(self.args, "block_size") else None
                )
                batch_gen = list(
                    batch_generater(
                        data,
                        observ_mask,
                        bs=self.args.batch_size,
                        n_nodes=self.args.n_nodes,
                        input_step=self.args.input_step,
                        pred_step=self.args.data_pred.pred_step,
                        block_size=block_size,
                    )
                )
                data_pred = data.clone()
                data_pred_all = data.clone()
                for x, y, t, mask_x, mask_y in batch_gen:
                    latent_pred_step += self.args.batch_size
                    y_pred, loss = self.latent_data_pred(x, y, mask_x, mask_y)
                    data_pred[t] = (
                        (y_pred * (1 - mask_y) + y * mask_y).clone().detach()[:, :, 0]
                    )
                    data_pred_all[t] = y_pred.clone().detach()[:, :, 0]
                self.data_pred_scheduler.step()
            if hasattr(self.args, "graph_discov"):
                for x, y, t, mask_x, mask_y in batch_gen:
                    graph_discov_step += self.args.batch_size
                    loss, loss_sparsity, loss_data = self.graph_discov(
                        x, y, mask_x, mask_y
                    )
                self.graph_scheduler.step()
                self.gumbel_tau *= self.gumbel_tau_gamma
                self.lambda_s *= self.lambda_gamma
            G_prob = self.G.detach().cpu().numpy()
            GT_prob = self.GT.detach().cpu().numpy()
            Graph = np.einsum("nm,ml->nl", G_prob, GT_prob)
        return Graph


def prepross_data(data):
    T, N, D = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(N):
        node = data[:, i, :]
        new_data[:, i, :] = (node - np.mean(node)) / np.std(node)
    return new_data


def main(data, mask, true_cm, opt, device="cuda"):
    if opt.n_nodes == "auto":
        opt.n_nodes = data.shape[-1]
    # Reshape data and mask from (T, n_nodes) to (T, n_nodes, 1)
    data = data[:, :, None]
    mask = mask[:, :, None]
    data = prepross_data(data)
    multicad = MultiCAD(opt, device=device)
    Graph = multicad.train(data, mask, data, true_cm)
    return Graph
