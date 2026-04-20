import torch
import torch.nn as nn
from .gru_ode_module import NNFODEwithBayesianJumps, SpatialGRU
from .gru_ode_utils import Block, DeepLabHead


class MMGRUODEPredictor(nn.Module):
    def __init__(self,
                 in_channels = 256,
                 latent_dim = 256,
                 num_gru_block = 2,
                 num_res_layers = 1,
                 mixture = True,
                 delta_t = 0.05):
        super(MMGRUODEPredictor, self).__init__()

        self.num_spatial_gru = num_gru_block
        self.delta_t = delta_t

        self.gru_ode = NNFODEwithBayesianJumps(
            input_size = in_channels,
            hidden_size= latent_dim,
            mixing = mixture,)


        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_spatial_gru):
            self.spatial_grus.append(SpatialGRU(in_channels, in_channels))
            if i < self.n_spatial_gru - 1:
                self.res_blocks.append(nn.Sequential(*[Block(in_channels) for _ in range(num_res_layers)]))
            else:
                self.res_blocks.append(DeepLabHead(in_channels, in_channels, 128))

        self.spatial_grus = nn.ModuleList(self.spatial_grus)
        self.res_blocks = nn.ModuleList(self.res_blocks)

    def forward(self,
                current_state, # current state [b, 1, c, h, w]
                camera_states, # history + current states of cameras [b, t, c, h, w]
                lidar_states, # history + current states of lidar [b, t, c, h, w]
                camera_timestamp, # history + current timestamp [b, t]
                lidar_timestamp, # history + current timestamp [b, t]
                target_timestamp): # maynot be used, as we don't need prediction

        # in multi-modal setting, t = 1
        x_bs = []
        bs, t, c, h, w = camera_states.shape

        for b in range(bs):
            obs_feats_w_time = dict()
            if camera_states is not None:
                 for id in range(t):
                    obs_feats_w_time[camera_timestamp[b, id].item()] = camera_states[b, id, :, :, ].unsqueeze(0)
            if lidar_states is not None:
                for id in range(t):
                    obs_feats_w_time[lidar_timestamp[b, id].item()] = lidar_states[b, id, :, :, ].unsqueeze(0)

            mm_obs_dict = dict(sorted(obs_feats_w_time.items(), key = lambda v: v[0]))

            times = torch.tensor(list(mm_obs_dict.keys()), device=current_state.device)
            observations = torch.stack(list(mm_obs_dict.values()), dim=1) # [1, t, c, h, w]

            updated_state, aux_loss, predict_state = self.gru_ode(
                times = times,
                input = current_state,
                obs = observations,
                delta_t = self.delta_t,
                T = target_timestamp[b],
            )
            x_bs.append(updated_state)

        x = torch.cat(x_bs, dim=0)

        ## is this step necessary? This is used for prediction...

        b, s, c, h, w = x.shape
        hidden_state = x[:, 0] # the first state in prediction? torch.Size([1, 64, 200, 200])
        for i in range(self.num_spatial_gru):
            x = self.spatial_grus[i](x, hidden_state)
            x = self.res_blocks[i](x.view(b * s, c, h, w))
            x = x.view(b, s, c , h, w)

        return x, aux_loss
