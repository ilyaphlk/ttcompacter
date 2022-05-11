"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch.nn as nn
import torch
from .adapter_utils import Activations
from seq2seq.hypercomplex.layers import PHMLinear
from .low_rank_layer import LowRankLinear
from t3nsor.layers import TTLinear, TTBias


class TensorTrainSingle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.cores_nonlinearity = None if config.cores_nonlinearity is None else Activations(config.cores_nonlinearity.lower())

        autoshapes = config.tt_shape is None
        self.tt_layer = TTLinear(
            self.input_dim, self.input_dim,
            bias=False,
            d=config.tt_d, tt_rank=config.tt_rank,
            shape=config.tt_shape, auto_shapes=autoshapes,
            auto_shape_mode=config.auto_shape_mode,
            reverse_out_shape=config.reverse_out_shape,
            use_scripted_mul=config.use_scripted_mul,
            cores_nonlinearity=self.cores_nonlinearity,
        )
        if config.use_bias and config.use_TTBias:
            self.bias = TTBias(self.input_dim, 1, c=1e-3, tt_rank=config.TTBias_rk)
        elif config.use_bias:
            self.bias = nn.Parameter(1e-3 * torch.ones(self.input_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.config.use_bias and self.config.use_TTBias:
            return self.bias(self.tt_layer(x))
        elif self.config.use_bias:
            return self.tt_layer(x) + self.bias
        return self.tt_layer(x)


class TensorTrainAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        if config.expansion_factor > 0:
            self.down_sample_size = self.input_dim * config.expansion_factor
        self.activation = Activations(config.non_linearity.lower())
        self.cores_nonlinearity = None if config.cores_nonlinearity is None else Activations(config.cores_nonlinearity.lower())
        if config.use_bias and config.use_TTBias:
            self.bias_down = TTBias(self.down_sample_size, 1, c=1e-3, tt_rank=config.TTBias_rk)
            self.bias_up = TTBias(self.input_dim, 1, c=1e-3, tt_rank=config.TTBias_rk)
        elif config.use_bias:
            self.bias_down = nn.Parameter(1e-3 * torch.ones(self.down_sample_size))
            self.bias_up = nn.Parameter(1e-3 * torch.ones(self.input_dim))
        else:
            self.register_parameter('bias_down', None)
            self.register_parameter('bias_up', None)

        autoshapes = config.tt_shape is None
        self.down_sampler = TTLinear(
            self.input_dim, self.down_sample_size,
            bias=False,
            d=config.tt_d, tt_rank=config.tt_rank,
            shape=config.tt_shape, auto_shapes=autoshapes,
            auto_shape_mode=config.auto_shape_mode,
            reverse_out_shape=config.reverse_out_shape,
            factorize_smaller_dim=config.factorize_smaller_dim,
            use_scripted_mul=config.use_scripted_mul,
            cores_nonlinearity=self.cores_nonlinearity,
        )

        self.tt_shape = self.down_sampler.shape
        upsample_shape = [list(reversed(self.tt_shape[1])), list(reversed(self.tt_shape[0]))]

        self.up_sampler = TTLinear(
            self.down_sample_size, self.input_dim,
            bias=False,
            d=config.tt_d, tt_rank=config.tt_rank,
            shape=upsample_shape, auto_shapes=autoshapes,
            auto_shape_mode=config.auto_shape_mode,
            reverse_out_shape=config.reverse_out_shape,
            factorize_smaller_dim=config.factorize_smaller_dim,
            use_scripted_mul=config.use_scripted_mul,
            cores_nonlinearity=self.cores_nonlinearity,
        )


    def forward(self, x):
        if self.config.use_bias and self.config.use_TTBias:
            z = self.bias_down(self.down_sampler(x))
        elif self.config.use_bias:
            z = self.down_sampler(x) + self.bias_down
        else:
            z = self.down_sampler(x)
        
        z = self.activation(z)

        if self.config.use_bias and self.config.use_TTBias:
            z = self.bias_up(self.up_sampler(z))
        elif self.config.use_bias:
            z = self.up_sampler(z) + self.bias_up
        else:
            z = self.up_sampler(z)
        return z


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim, 
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)
    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)