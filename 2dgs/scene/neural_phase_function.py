import math

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    class _FrequencyEncoding(nn.Module):
        def __init__(self, n_input_dims, encoding_config):
            super().__init__()
            self.n_input_dims = n_input_dims
            self.n_frequencies = int(encoding_config["n_frequencies"])
            self.n_output_dims = self.n_input_dims * self.n_frequencies * 2

        def forward(self, x):
            outputs = []
            for i in range(self.n_frequencies):
                freq = (2.0 ** i) * math.pi
                outputs.append(torch.sin(freq * x))
                outputs.append(torch.cos(freq * x))
            return torch.cat(outputs, dim=-1)

    class _TinyLikeNetwork(nn.Module):
        def __init__(self, n_input_dims, n_output_dims, network_config):
            super().__init__()
            hidden_dim = int(network_config["n_neurons"])
            hidden_layers = int(network_config["n_hidden_layers"])
            activation = nn.LeakyReLU if network_config.get("activation") == "LeakyReLU" else nn.ReLU
            output_activation_name = network_config.get("output_activation")
            output_activation = nn.Sigmoid if output_activation_name == "Sigmoid" else nn.Identity

            layers = []
            in_dim = n_input_dims
            for _ in range(hidden_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(activation())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_output_dims))
            layers.append(output_activation())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class _TinyCudaNNFallback:
        Encoding = _FrequencyEncoding
        Network = _TinyLikeNetwork

    tcnn = _TinyCudaNNFallback()


class Neural_phase(nn.Module):
    def __init__(self, hidden_feature_size=32, hidden_feature_layers=3, frequency=4, neural_material_size=6, asg_mlp=False):
        super().__init__()
        self.neural_material_size = neural_material_size
        self.asg_mlp = asg_mlp

        encoding_config = {"otype": "Frequency", "n_frequencies": frequency}
        shadow_config = {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "Sigmoid",
            "n_neurons": hidden_feature_size,
            "n_hidden_layers": hidden_feature_layers,
        }
        other_effects_config = {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "Sigmoid",
            "n_neurons": 128,
            "n_hidden_layers": 3,
        }
        asg_func_config = {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "Sigmoid",
            "n_neurons": 32,
            "n_hidden_layers": 3,
        }

        self.encoding = tcnn.Encoding(3, encoding_config)
        phase_input_dim = self.neural_material_size + self.encoding.n_output_dims * 3 + 1
        effects_input_dim = self.neural_material_size + self.encoding.n_output_dims * 3
        self.shadow_func = tcnn.Network(phase_input_dim, 1, shadow_config)
        self.other_effects_func = tcnn.Network(effects_input_dim, 3, other_effects_config)
        if self.asg_mlp:
            self.asg_func = tcnn.Network(self.neural_material_size + 3, 1, asg_func_config)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, wi, wo, pos, neural_material, encoding_pos=None, hint=None, asg_1=None, asg_mlp=False):
        if encoding_pos is None:
            wi_enc = self.encoding(wi)
            wo_enc = self.encoding(wo)
            pos_enc = self.encoding(pos)
            encoding_pos = {"wi_enc": wi_enc, "wo_enc": wo_enc, "pos_enc": pos_enc}
        else:
            wi_enc = encoding_pos["wi_enc"]
            wo_enc = encoding_pos["wo_enc"]
            pos_enc = encoding_pos["pos_enc"]

        other_effects = self.other_effects_func(torch.cat([wo_enc, pos_enc, wi_enc, neural_material], dim=-1)) * 2.0
        decay = None
        if hint is not None:
            decay = self.shadow_func(torch.cat([wo_enc, wi_enc, pos_enc, neural_material, hint], dim=-1))
            decay = torch.relu(decay - 1e-5)

        if asg_mlp and asg_1 is not None:
            asg_3 = self.asg_func(torch.cat([neural_material, asg_1], dim=-1))
        else:
            asg_3 = asg_1

        return decay, other_effects, asg_3, encoding_pos
