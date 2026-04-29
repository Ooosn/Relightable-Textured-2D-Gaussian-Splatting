from collections import OrderedDict
import torch
import torch.nn as nn
import math

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
            activation_name = network_config.get("activation", "ReLU")
            output_activation_name = network_config.get("output_activation", None)

            activation = nn.LeakyReLU if activation_name == "LeakyReLU" else nn.ReLU
            if output_activation_name == "Sigmoid":
                output_activation = nn.Sigmoid
            else:
                output_activation = nn.Identity

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


"""
- tcnn.Encoding（输入特征编码）
- tcnn.Network（神经网络结构）
- tcnn.Optimizer（优化器）
- tcnn.Loss（损失函数）
- tcnn.Linear（高效的全连接层）

    tcnn 默认没有 bias
    otype 模型类型


"""
# Neural_phase 继承了 torch.nn.Module，成为一个 PyTorch 的可训练模型
# 两要素：1. __init__ 构造神经网络，2. forward 函数，用于定义前向传播
class Neural_phase(nn.Module):
    def __init__(self, hidden_feature_size=32, hidden_feature_layers=3, frequency=4, neural_material_size=6, asg_mlp=False, mlp_zero=False):
        super().__init__()
        self.neural_material_size = neural_material_size
        self.asg_mlp = asg_mlp
        
        # 利用 tcnn 创建更高效的 mlp
        # configuration
        # frequency encoding
        encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": frequency
                }
        # shadow refine
        shadow_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": hidden_feature_size,
                    "n_hidden_layers": hidden_feature_layers,
                }
        # other effects
        other_effects_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": 128,
                    "n_hidden_layers": 3,
                }
        
        # asg_func
        asg_func_config = {
                        "otype": "FullyFusedMLP",
                        "activation": "LeakyReLU",
                        "output_activation": "Sigmoid", 
                        "n_neurons": 32,
                        "n_hidden_layers": 3,
                }
            
        # 位置编码：
        # 高频特征映射，把低维的 3D 坐标映射到一个高维的特征空间，让神经网络更容易学习高频细节
        # 4-band positional encoding，文中将 frequency 设置为4
        self.encoding = tcnn.Encoding(3, encoding_config)

        # 修正阴影值：
        # 输入维度为 高斯点的材质维度/高斯点的可学习隐变量（neural_material_size）+ 光源方向，高斯中心高频特征（encoding.n_output_dims * 2） + 原始阴影维度（1）
        # 输出为 修正阴影值（维度为1）
        self.shadow_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 3 + 1, 1, shadow_config)
        # self.shadow_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 2 + 1, 1, shadow_config)

        # 其他效果：
        # 输入维度为 高斯点的材质维度/高斯点的可学习隐变量（neural_material_size）+ 观察方向、高斯中心高频特征 (encoding.n_output_dims * 2）
        # 输出为 RGB 分量，RPG components
        self.other_effects_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 3, 3, other_effects_config)

        # 修正高光通道：
        # 输入维度为 高斯点的材质维度/高斯点的可学习隐变量（neural_material_size）+ asg_1 的特征维度
        # 输出为 asg_1 的修正
        if self.asg_mlp:
            self.asg_func = tcnn.Network(self.neural_material_size + 3, 1, asg_func_config)
        
        """
        在 PyTorch 中，named_children() 返回当前模型的所有子模块的名称和对象，名称一般是模块的顺序
        """
        # 将所有线性层的参数初始化为 0.0
        # 每个高斯点都是独立的个体，MLP 初始时 不应该对数据进行过早的偏向（bias），而是应该让它们保持一致，等待训练逐步调整。
        # 采用 leaky_relu 作为激活函数，可以防止梯度消失，允许参数更快的跳出0，减少 weight=0 的影响
        

        # 遗留代码，没用的，tinycudann 和 nnmodules 不一样, tinycudann 不使用 nn.module 的模块，所以 named_children() 为空
        # 而且如果真的全是 0 ，由于多重 leaky_relu 的缘故，会导致梯度消失，无法训练
        if False:
            for n, m in self.shadow_func.named_children():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight.data, 0.0)
                    
            for n, m in self.other_effects_func.named_children():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight.data, 0.0)

            for n, m in self.asg_func.named_children():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight.data, 0.0)
        
        # 真正的 初始为 0 , 不好用
        if mlp_zero:
            print("mlp_zero")
            with torch.no_grad():
                for param in self.asg_func.parameters():
                    param.uniform_(-0.01, 0.01)  # 小随机值
                for param in self.shadow_func.parameters():
                    param.uniform_(-0.01, 0.01)
                for param in self.other_effects_func.parameters():
                    param.uniform_(-0.01, 0.01)

       

    """
    通过设置 requires_grad 来设置冻结和解冻，不会自动初始化 grad 为 0 ， 仍会保留之前的梯度！
    """
    def freeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = False
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = False
        if self.asg_mlp:
            for name, param in self.asg_func.named_parameters():
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = True
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = True
        if self.asg_mlp:
            for name, param in self.asg_func.named_parameters():
                param.requires_grad = True


    def forward(self, wi, wo, pos, neural_material, encoding_pos = None, opacity = None, hint = None, asg_1 = None, asg_mlp = False):
        # 光源方向
        if encoding_pos is None:
            wi_enc = self.encoding(wi)
            # 观察方向
            wo_enc = self.encoding(wo)
            # 高斯中心
            pos_enc = self.encoding(pos)

            encoding_pos = {"wi_enc": wi_enc, "wo_enc": wo_enc, "pos_enc": pos_enc}

        else:
            wi_enc = encoding_pos["wi_enc"]
            wo_enc = encoding_pos["wo_enc"]
            pos_enc = encoding_pos["pos_enc"]

        # hint：为原阴影值-shadow，而decay 为修正阴影值
        # other_effects，RGB 颜色修正（用于考虑其他效果带来的影响）
        # asg_1：原高光通道，channel_num = 1，修正后 channel_num = 3
        
        """
        1. torch.concat = torch.cat
            dim = -1 : 按最后一个维度拼接，如果是二维，dim = 0，也就是按行扩展，dim = 1，也就是 dim = -1 就是按列扩展
        """
        # 所以这里扩展后，每行对应各个高斯点，如果是其他模型，对应也就是 batch 中每个数据，每列对应每个高斯点的特征向量，这里拼接了 光源方向，高斯点中心，高斯点可学习的隐变量
        if opacity is None:
            if hint is not None:
                bad_inputs = []
                for name, tensor in (
                    ("wi_enc", wi_enc),
                    ("wo_enc", wo_enc),
                    ("pos_enc", pos_enc),
                    ("neural_material", neural_material),
                    ("hint", hint),
                ):
                    if tensor is not None and not torch.isfinite(tensor).all():
                        bad_inputs.append(name)
                bad_params = []
                for name, param in self.other_effects_func.named_parameters():
                    if not torch.isfinite(param).all():
                        bad_params.append(name)
                if bad_inputs or bad_params:
                    raise FloatingPointError(
                        "Non-finite input before other_effects_func: "
                        f"bad_inputs={bad_inputs}, bad_other_effects_params={bad_params}"
                    )

            other_effects = self.other_effects_func(torch.concat([wo_enc, pos_enc, wi_enc, neural_material], dim=-1)) * 2.0  #用于后面变为 [-1，1] 的颜色范围
            """
            1. torch.isnan(tensor): 返回一个 布尔张量，其中 True 表示该位置的值是 NaN
            2. any(): 检查张量中是否 至少有一个 True：如果 tensor 中至少有一个 True，返回 True
            3. assert：assert 语句用于 程序的调试和检查，如果条件为 False，程序会抛出 AssertionError 并终止运行。
            4. 任何数和 NaN 进行数学运算，结果仍然是 NaN，最终可能污染输出，loss，梯度，以至于其他参数
            """
            
            # 当 other_effects 有 NaN 值，会返回包含 True 值的张量，随后在 any() 函数下返回 True，经过 not 变为 False，最终由于 assert False ， 抛出 AssertionError 异常，终止运行。
            if hint is not None:
                if not torch.isfinite(other_effects).all():
                    raise FloatingPointError(
                        "other_effects_func produced non-finite outputs; "
                        "check wi/wo/pos/neural_material and the MLP weights."
                    )
                if hint.shape[1] == 2:
                    hint_reshaped = torch.cat([hint[:, 0:1], hint[:, 1:2]], dim=0)
                    wi_enc = wi_enc.repeat(2, 1)
                    pos_enc = pos_enc.repeat(2, 1)
                    neural_material = neural_material.repeat(2, 1)
                else:
                    hint_reshaped = hint
                output1 = self.shadow_func(torch.concat([wo_enc, wi_enc, pos_enc, neural_material, hint_reshaped], dim=-1))
                output1 = torch.relu(output1 - 1e-5)
            else:
                output1 = None
            # decay = self.shadow_func(torch.concat([wi_enc, pos_enc, neural_material, hint], dim=-1))
            """
            torch.relu(tensor) 对张量进行逐元素 relu 操作（也就是再次经过 relu 激活函数），大于 0 的值维持原值，小于 0 的值取 0
            """

            # torch.relu(decay - 1e-5) 如果 decay 过小，小于 1e-5，则设置为 0（因为相减后会为负）。
            # 这里应该除了为了减少计算开销，同时因为用的 leaky_relu 最后结果可能为负，所以需要用 relu 修正
            # decay = hint * decay
            
        # 修正透明度
        else:
            opacity = opacity.detach()
            wi_enc = wi_enc.detach()
            wo_enc = wo_enc.detach()
            pos_enc = pos_enc.detach()
            output1 = self.shadow_func(torch.concat([wo_enc, wi_enc, pos_enc, neural_material, opacity], dim=-1))
            encoding_pos = {"wi_enc": wi_enc, "wo_enc": wo_enc, "pos_enc": pos_enc}
            other_effects = None
            output1 = torch.relu(output1 - 1e-5)

        # 修正高光通道
        if asg_mlp:
            asg_3 = self.asg_func(torch.concat([neural_material, asg_1], dim=-1))
        else:
            asg_3 = asg_1
            

        return output1, other_effects, asg_3, encoding_pos
