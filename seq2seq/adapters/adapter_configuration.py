"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import torch.nn as nn

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    task_expansion_factor: int = 0
    add_adapter_in_feed_forward: bool = True
    add_adapter_in_self_attention: bool = True
    hidden_dim: int = 128
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder: bool = True
    intrinsic_dim: int = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.01

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1


    # Tensor-Train adapters.
    tensor_train_adapters: bool = False
    tensor_train_single: bool = False
    tt_rank: int = 8
    tt_d: int = 3
    tt_shape: List[List[int]] = None
    reverse_out_shape: bool = False
    cores_nonlinearity: str = None
    use_scripted_mul: bool = False
    auto_shape_mode: str = 'ascending'
    naive: bool = False

    use_checkpointing: bool = False
    ttcore_checkpointing: bool = False

    use_ScaleNorm: bool = False
    ScaleNorm_scale: float = 1.0
    use_TTLayerNorm: bool = False
    TTLayerNorm_rk: int = 2
    TTLayerNorm_preinit: bool = False
    use_LayerNorm_mean: bool = False
    use_bias: bool = True
    use_TTBias: bool = False
    TTBias_rk: int = 2

    use_LoRA: bool = False
    lora_dense: bool = False
    use_TTLoRA: bool = False
    ttlora_separate: bool = False
    TTLoRA_init: str = None

    factorize_smaller_dim: bool = True
    freeze_cores = None  # either None, 'first' or 'last'



ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
