import torch
import torch.nn as nn
import numpy as np
import math

from .gru_ode_utils import ConvNet, Bottleblock, SmallEncoder, SmallDecoder, rsample_normal

class NNFODEwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self,
                 input_size,
                 hidden_size,
                 # cfg,
                 bias = True,
                 logvar = True,
                 mixing = 1,
                 solver = "euler",
                 min_log_sigma = -5.0,
                 max_log_sigma = 5.0,
                 impute = False,
                 skipco = False,
                 use_variable_ode_step = False,
                 encoder_out_channels = 64,
                 encoder_filter_size = 64):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super().__init__()

        self.impute = impute
        # self.cfg = cfg

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.p_model = ConvNet(hidden_size, hidden_size * 2, )

        self.gru_c = DualGRUODECell(input_size,
                                    hidden_size,
                                    bias=bias)

        self.gru_obs = GRUObservationCell(input_size,
                                          hidden_size,
                                          min_log_sigma=self.min_log_sigma,
                                          max_log_sigma=self.max_log_sigma,
                                          bias=bias)
        self.skipco = skipco

        self.srvp_encoder = SmallEncoder(encoder_out_channels,
                                         encoder_out_channels,
                                         encoder_filter_size,)
        self.srvp_decoder = SmallDecoder(encoder_out_channels,
                                         encoder_out_channels,
                                         encoder_filter_size,
                                         self.skipco)

        self.solver = solver,
        self.use_variable_ode_step = use_variable_ode_step
        assert self.solver in ["euler", "midpoint"], f"Solver must be either 'euler' or 'midpoint', but got {self.solver}"

        # self.store_hist = options.pop("store_hist",False)
        self.input_size = input_size
        self.logvar = logvar
        self.mixing = mixing  # mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def srvp_decode(self, x, skip=None):
        # content vector is missing.
        """
        Decodes SRVP intermediate states into LSS output space
        x: SRVP intermediate states, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
        torch.Tensor: [batch, seq_len, channels, height, width]
        """
        b, t, c, h, w = x.shape
        _x = x.reshape(b * t, c, h, w)
        if skip:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])

    def srvp_encode(self, x):
        """
        Encodes LSS's outputs
            x: LSS encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
            torch.Tensor: [batch, seq_len, channels, height, width]
        """
        b, t, c, h, w = x.shape
        _x = x.view(b * t, c, h, w)
        hx, skips = self.srvp_encoder(_x, return_skip=True)
        hx = hx.view(b, t, *hx.shape[1:])
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                tx = torch.randint(t, size=(b,)).to(hx.device)
                index = torch.arange(b).to(hx.device)
                skips = [s.view(b, t, *s.shape[1:])[index, tx] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(b, t, *s.shape[1:])[:, -1] for s in skips]
        else:
            skips = None
        return hx, skips

    def ode_step(self, state, input, delta_t, current_time):

        """Executes a single ODE step."""
        eval_times = torch.tensor([0], device=state.device, dtype=torch.float64)
        eval_ps = torch.tensor([0], device=state.device, dtype=torch.float32)

        if self.impute is False:
            input = torch.zeros_like(input)

        if self.solver == "euler":
            state = state + delta_t * self.gru_c(input, state)
            input = self.infer_state(state)[0]
        elif self.solver == "midpoint":
            k = state + delta_t / 2 * self.gru_c(input, state)
            pk = self.infer_state(k)[0]

            state = state + delta_t * self.gru_c(pk, k)
            input = self.infer_state(state)[0]
        else:
            raise ValueError(f"Unknown solver '{self.solver}'.")

        current_time += delta_t
        return state, input, current_time, eval_times, eval_ps

    def infer_state(self, x, deterministic=False):
        """
        Creates the first state from the conditioning observation
            x: encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
            deterministic: If true, it will return only the means otherwise a sample from Normal distribution
        Returns:
            First state of the model -> y, torch.Tensor, [batch, channels, height, width]
        """
        # Q1: will the first state be stochastic?
        # Q2: are we going to sample a different noise for each position

        q_y0_params = self.p_model(x)
        y_0 = rsample_normal(q_y0_params,
                             max_log_sigma=self.max_log_sigma,
                             min_log_sigma=self.min_log_sigma)
        return y_0, q_y0_params

    def forward(self,
                times,
                input,
                obs,
                delta_t,
                T,
                return_path=True, ):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of state

        Returns:
            state          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        # state = self.covariates_map(cov)

        # p            = self.p_model(state)
        # vis_bev_feature(obs,'obs')

        hx_obs, skips_obs = self.srvp_encode(obs)
        input, input_obs = self.srvp_encode(input)
        bs, seq, c, h, w = input.shape  # [1, 1, 64, 200, 200]
        input = input.view(bs * seq, c, h, w)

        state = torch.zeros_like(input)  # constant init of temporal state
        current_time = times.min().item()

        # counter = 0
        #
        # loss_pre_jump = 0  # Pre-jump loss
        # loss_post_jump = 0  # Post-jump loss (KL between p_updated and the actual sample)

        path_t = []
        path_h = []
        # path_p = []
        # if return_path:
        #     path_t = [0]
        #     path_p = [input]
        #     path_h = [state]

        # if smoother:
        #     class_loss_vec = torch.zeros(cov.shape[0],device = state.device)
        #     num_evals_vec  = torch.zeros(cov.shape[0],device = state.device)
        #     class_criterion = class_criterion
        #     assert class_criterion is not None

        # assert len(times) + 1 == len(time_ptr)

        # assert (len(times) == 0) or (times[-1] <= max(T))

        eval_times_total = torch.tensor([], dtype=torch.float64, device=state.device)
        eval_vals_total = torch.tensor([], dtype=torch.float32, device=state.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time <= (obs_time - delta_t):  # 0.0001 delta_t used for numerical consistency.
                if self.use_variable_ode_step:
                    state, input, current_time, eval_times, eval_ps = self.ode_step(state, input,
                                                                                    obs_time - current_time,
                                                                                    current_time)
                else:
                    state, input, current_time, eval_times, eval_ps = self.ode_step(state, input, delta_t,
                                                                                    current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()
                # if current_time < obs_time:
                # Storing the predictions.
                # if return_path:
                #     path_t.append(current_time)
                #     path_p.append(input)
                #     path_h.append(state)

            ## Reached an observation
            X_obs = hx_obs[:, i, :, :, :]

            ## Using GRUObservationCell to update state. Also updating input and loss
            state, losses = self.gru_obs(state, input, X_obs)

            # if smoother:
            #     class_loss_vec[i_obs] += class_criterion(self.classification_model(state[i_obs]),labels[i_obs]).squeeze(1)
            #     num_evals_vec[i_obs] +=1
            # if losses.sum()!=losses.sum():

            # loss_pre_jump    = loss_pre_jump+ losses.sum()

            input = self.infer_state(state)[0]

            # loss_post_jump = loss_post_jump + compute_KL_loss(p_obs = input, X_obs = X_obs, logvar=self.logvar)

            if return_path:
                path_t.append(obs_time.item())
                # path_p.append(input)
                path_h.append(state)

        # current_time = 1.0
        ## after every observation has been processed, propagating until T
        for predict_time in T:
            while current_time < predict_time:
                if self.solver == "dopri5":
                    state, input, current_time, eval_times, eval_ps = self.ode_step(state, input,
                                                                                    predict_time - current_time,
                                                                                    current_time)
                else:
                    if self.use_variable_ode_step:
                        state, input, current_time, eval_times, eval_ps = self.ode_step(state, input,
                                                                                        predict_time - current_time,
                                                                                        current_time)
                    else:
                        state, input, current_time, eval_times, eval_ps = self.ode_step(state, input, delta_t,
                                                                                        current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                # counter += 1
                # current_time = counter * delta_t
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()
                # Storing the predictions
                if current_time > predict_time - 0.5 * delta_t and current_time < predict_time + 0.5 * delta_t:
                    path_t.append(current_time)
                    # path_p.append(input)
                    path_h.append(state)

        x = []
        path_t = np.array(path_t)

        for time_stamp in T:
            if isinstance(time_stamp, torch.Tensor):
                time_stamp = time_stamp.item()
            A = np.where(path_t > time_stamp - 0.5 * delta_t)[0]
            B = np.where(path_t < time_stamp + 0.5 * delta_t)[0]

            if np.any(np.in1d(A, B)):
                idx = np.max(A[np.in1d(A, B)])
            else:
                idx = np.argmin(np.abs(path_t - time_stamp))
            x.append(path_h[idx])

        x = torch.stack(x, dim=1)

        x = self.srvp_decode(x)

        loss = 0
        return state, loss, x



class DualGRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_update_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.trusting_gate = nn.Sequential(
            Bottleblock(hidden_size + hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
        )

    def forward(self, x, state):
        '''
        x: torch.Tensor [b, 1, input_size, h, w]
        state: torch.Tensor [b, n_present, hidden_size, h, w]
        '''
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
            state = state.unsqueeze(0)
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]

        # warmup
        for t in range(n_present - 1):
            cur_state = state[:, t]
            h = self.gru_cell_2(cur_state, h)

        # recurrent layers
        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        # propagate gru v1
        rnn_state1 = self.gru_cell_1(x, rnn_state1)
        # propagate gru v2
        h = self.gru_cell_2(rnn_state2, h)
        rnn_state2 = self.conv_decoder_2(h)

        # mix the two distribution
        mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
        trust_gate = self.trusting_gate(mix_state)
        trust_gate = torch.softmax(trust_gate, dim=1)
        cur_state = rnn_state2 * trust_gate[:, 0:1] + rnn_state1 * trust_gate[:, 1:]

        return cur_state - state.squeeze(1)

    def gru_cell_1(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, min_log_sigma=-5.0, max_log_sigma=5.0, bias=True):
        super().__init__()
        self.gru_d = DualGRUCell(input_size, hidden_size, bias=bias)
        prep_hidden = hidden_size
        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def forward(self, state, p, X_obs):
        bs, C, h, w = X_obs.shape

        # mean = mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)
        # var = var.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)
        # error = error.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h,w)

        gru_input = X_obs

        state = self.gru_d(gru_input, state)

        loss = None
        return state, loss


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional previous state and passes a
    convolutional gated recurrent unit over the data"""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder = nn.Conv2d(hidden_size, input_size, kernel_size=1, bias=False)

    def forward(self, x, state=None):
        # pylint: disable=unused-argument, arguments-differ
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'

        # recurrent layers
        rnn_output = []
        b, timesteps, c, h, w = x.size()
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]

            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(self.conv_decoder(rnn_state))

        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output