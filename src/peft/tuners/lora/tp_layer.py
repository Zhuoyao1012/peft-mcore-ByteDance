# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import importlib
import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose
from peft.utils.integrations import gather_params_ctx

from .layer import LoraLayer


class LoraParallelLinear(nn.Module, LoraLayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        backend,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self.backend = backend
        self.is_parallel_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method
        input_is_parallel = True
        gather_output = False
        if isinstance(base_layer, self.backend.RowParallelLinear):
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output,
            **parallel_linear_kwargs,
        )

        if is_target_conv_1d_layer:
            raise ValueError(
                f"{self.__class__.__name__} does not support target_conv_1d_layer yet, please set it to False"
            )
        self.is_target_conv_1d_layer = False

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora=False,
        init_method=init.xavier_normal_,
        input_is_parallel=True,
        gather_output=False,
        **parallel_linear_kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        megatron_config = parallel_linear_kwargs["megatron_config"]
        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        megatron_config.params_dtype = torch.float32
        if self.is_parallel_a:
            lora_a = self.backend.RowParallelLinear(
                input_size=self.in_features,
                output_size=r,
                bias=False,
                input_is_parallel=input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
                config=megatron_config,
            )
            lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
        else:
            lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            lora_b = self.backend.ColumnParallelLinear(
                input_size=r,
                output_size=self.out_features,
                bias=False,
                gather_output=gather_output,
                init_method=init_method,
                config=megatron_config,
            )
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)
        return result, bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

def pad_seq_to_mult(x, mult):
    import torch.nn.functional as F

    if x.shape[0] % mult == 0:
        return x, 0
    pad_len = mult - (x.shape[0] % mult)
    with torch.no_grad():
        # pad at the tail
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    return x, pad_len

def unpad_seq_to_mult(x, pad_len):
    if pad_len <= 0:
        return x
    with torch.no_grad():
        # prune tail padding
        return x[:-pad_len, :]

class LoraTEParallelLinear(nn.Module, LoraLayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        te_backend,
        backend,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self.backend = backend
        self.te_backend = te_backend
        self.input_is_parallel = isinstance(base_layer, te_backend.TERowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        self.update_inout_dims(megatron_config)
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method
        self.is_expert = getattr(self.base_layer, 'is_expert', False)

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_method=init_method,
            **parallel_linear_kwargs,
        )

        if is_target_conv_1d_layer:
            raise ValueError(
                f"{self.__class__.__name__} does not support target_conv_1d_layer yet, please set it to False"
            )
        self.is_target_conv_1d_layer = False

    def update_inout_dims(self, megatron_config):
        tp_size = megatron_config.tensor_model_parallel_size
        if isinstance(self.base_layer, self.te_backend.TEColumnParallelLinear) or \
            isinstance(self.base_layer, self.te_backend.TELayerNormColumnParallelLinear):
            self.input_is_parallel = False
            # m.in_features and m.out_features are divided by tp_size already,
            # but in_features and out_features passed to ParallelLinearAdapter are not.
            self.in_features = self.base_layer.in_features
            self.out_features = self.base_layer.out_features * tp_size
            if isinstance(self.base_layer, self.te_backend.TELayerNormColumnParallelLinear):
                # LoRA is applied after layernorm, so layernorm output must be returned
                self.base_layer.return_layernorm_output = True
                # perf optimization for LoRA + SP
                if self.base_layer.config.sequence_parallel and not getattr(self.base_layer,'ub_overlap_ag', False):
                    self.base_layer.return_layernorm_output_gathered = True
        elif isinstance(self.base_layer, self.te_backend.TERowParallelLinear):
            self.input_is_parallel = True
            self.in_features = self.base_layer.in_features * tp_size
            self.out_features = self.base_layer.out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora=False,
        init_method=init.xavier_normal_,
        **parallel_linear_kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        megatron_config = parallel_linear_kwargs["megatron_config"]
        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        # megatron_config.params_dtype = torch.float32

        self._sequence_parallel = megatron_config.sequence_parallel
        megatron_config.sequence_parallel = False  # SP is irrelevant for the lora linear layer
        self.config = megatron_config 

        if self.input_is_parallel:
            lora_a = self.backend.RowParallelLinear(
                self.in_features,
                r,
                config=megatron_config,
                input_is_parallel=True,
                skip_bias_add=True,
                bias=False,
                init_method=init_method,
            )
        else:
            lora_a = self.backend.ColumnParallelLinear(
                self.in_features,
                r,
                config=megatron_config,
                bias=False,
                gather_output=True,
                init_method=init_method,
                disable_grad_reduce=self._sequence_parallel,
            )

        # (@adithyare) we use this option to mirror the behavior a column parallel layer with two low-rank column parallel layers
        # if the original column parallel layer uses gather_output=False, then we will use the self.liner_out layer defined below.
        lin_out_gather_output = True if self.input_is_parallel else False
        lora_b = self.backend.ColumnParallelLinear(
            r,
            self.out_features,
            config=megatron_config,
            bias=False,
            gather_output=lin_out_gather_output,
            init_method=init_method,
        )

        # cast all parameters when using amp O2 training
        if megatron_config.bf16:
            lora_a.bfloat16()
            lora_b.bfloat16()
        elif megatron_config.fp16:
            lora_a.half()
            lora_b.half()

        # revert config change in case it is read elsewhere
        megatron_config.sequence_parallel = self._sequence_parallel
        if self._sequence_parallel and not self.input_is_parallel:
            from importlib.metadata import version

            import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("1.5.0dev") and (
                isinstance(self.base_layer, self.te_backend.TELayerNormColumnParallelLinear)
                and (
                    not getattr(megatron_config, "tp_comm_overlap", False)
                    or getattr(megatron_config, "tp_comm_overlap_disable_qkv", False)
                )
            ):
                # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                # in the forward method is not needed, so set self._sequence_parallel to False
                # unless TP communication overlap is used
                self._sequence_parallel = False

        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def _base_layer_fwd(self, x, *args, **kwargs):
        linear_output = self.base_layer(x, *args, **kwargs)
        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        """ Four cases for the wrapped module's return values
        1. nothing: (out, None)
        2. return_bias: (out, bias)
        2. return_layernorm_output: ((out, ln_out), None)
        3. both: (out, bias, ln_out)
        """
        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
                x = layernorm_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output
            x = layernorm_output

        # return linear_output and adapter input
        return linear_output, x.contiguous(), bias

    def _lora_fwd(self, x, active_adapter):
        gather_for_sp = self.backend.mappings.gather_from_sequence_parallel_region
        scatter_for_sp = self.backend.mappings.scatter_to_sequence_parallel_region

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)

        x = dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.tensor_model_parallel_size)

        if self._sequence_parallel and not self.input_is_parallel:
            # for attention_qkv and linear_fc1
            # layernorm before lora is impacted by sequence parallel,
            # hence seq dim need to be gathered right before lora linear layers
            # this function also handles the backward pass correctly
            x = gather_for_sp(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = lora_A(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = lora_B(x)

        if self._sequence_parallel and self.input_is_parallel and not self.is_expert:
            x = scatter_for_sp(x)

        x = x * scaling

        if pad_len > 0:
            # Remove MoE padding.
            x = unpad_seq_to_mult(x, pad_len)

        return x


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            result, _, bias = self._base_layer_fwd(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
        elif self.merged:
            assert False, "can not support weight merge for now."
        else:
            result, adapter_in, bias = self._base_layer_fwd(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                result = result + self._lora_fwd(adapter_in, active_adapter)
            result = result.to(torch_result_dtype)
        return result, bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        assert False, "Cannot support merge for now."
        # adapter_names = check_adapters_to_merge(self, adapter_names)
        # if not adapter_names:
        #     # no adapter to merge
        #     return

        # for active_adapter in adapter_names:
        #     if active_adapter in self.lora_A.keys():
        #         base_layer = self.get_base_layer()
        #         if safe_merge:
        #             # Note that safe_merge will be slower than the normal merge
        #             # because of the copy operation.
        #             orig_weights = base_layer.weight.data.clone()
        #             delta_weight = self.get_delta_weight(active_adapter)
        #             if not self.use_dora[active_adapter]:
        #                 orig_weights = orig_weights + delta_weight
        #             else:
        #                 # handle dora
        #                 # since delta_weight already includes scaling, set it to 1 here
        #                 weight_norm = (
        #                     self.lora_magnitude_vector[active_adapter]
        #                     .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
        #                     .detach()
        #                 )
        #                 # We need to cache weight_norm because it has to be based on the original weights. We
        #                 # cannot calculate it on the fly based on the merged weights when unmerging because its a
        #                 # different value
        #                 self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        #                 dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
        #                 dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
        #                 orig_weights = dora_factor * (orig_weights + delta_weight)

        #             if not torch.isfinite(orig_weights).all():
        #                 raise ValueError(
        #                     f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
        #                 )

        #             base_layer.weight.data = orig_weights
        #         else:
        #             delta_weight = self.get_delta_weight(active_adapter)
        #             if not self.use_dora[active_adapter]:
        #                 base_layer.weight.data = base_layer.weight.data + delta_weight
        #             else:
        #                 # handle dora
        #                 # since delta_weight already includes scaling, set it to 1 here
        #                 weight_norm = (
        #                     self.lora_magnitude_vector[active_adapter]
        #                     .get_weight_norm(
        #                         base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
        #                     )
        #                     .detach()
        #                 )
        #                 # We need to cache weight_norm because it has to be based on the original weights. We
        #                 # cannot calculate it on the fly based on the merged weights when unmerging because its a
        #                 # different value
        #                 self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        #                 dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
        #                 dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
        #                 new_weight = dora_factor * (base_layer.weight.data + delta_weight)
        #                 base_layer.weight.data = new_weight

        #         self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        assert not self.merged, "This layer should not be merge for not supported."
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        # while len(self.merged_adapters) > 0:
        #     active_adapter = self.merged_adapters.pop()
        #     if active_adapter in self.lora_A.keys():
        #         weight = self.get_base_layer().weight
        #         delta_weight = self.get_delta_weight(active_adapter)
        #         if not self.use_dora[active_adapter]:
        #             weight.data -= delta_weight
        #         else:
        #             weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
        #             dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
        #             weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
        #             weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        pass
        # device = self.lora_B[adapter].weight.device
        # dtype = self.lora_B[adapter].weight.dtype

        # # In case users wants to merge the adapter weights that are in
        # # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        # cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        # weight_A = self.lora_A[adapter].weight
        # weight_B = self.lora_B[adapter].weight

        # if cast_to_fp32:
        #     weight_A = weight_A.float()
        #     weight_B = weight_B.float()

        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        # if cast_to_fp32:
        #     output_tensor = output_tensor.to(dtype=dtype)

        #     # cast back the weights
        #     self.lora_A[adapter].weight.data = weight_A.to(dtype)
        #     self.lora_B[adapter].weight.data = weight_B.to(dtype)

        # return output_tensor

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ):
        """For mcore to save distributed checkpoints."""
        lora_a = self.lora_A[self._active_adapter]
        lora_b = self.lora_B[self._active_adapter]
        sharded_state_dict = {}
        sharded_state_dict.update(lora_a.sharded_state_dict(f"{prefix}lora_A.", sharded_offsets, metadata))
        sharded_state_dict.update(
            lora_b.sharded_state_dict(f"{prefix}lora_B.", sharded_offsets, metadata)
        )
        return sharded_state_dict


def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None

    if megatron_core:
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs["megatron_config"] = megatron_config
        if megatron_kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` "
                "or `RowParallelLinear`. "
                "Setting fan_in_fan_out to False."
            )
            megatron_kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False

        if isinstance(
            target_base_layer,
            (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear,),
        ):

            new_module = LoraParallelLinear(
                base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs
            )
        elif isinstance(
            target_base_layer, 
            (megatron_core.extensions.transformer_engine.TELayerNormColumnParallelLinear, 
            megatron_core.extensions.transformer_engine.TEColumnParallelLinear, 
            megatron_core.extensions.transformer_engine.TERowParallelLinear,),
        ):
            new_module = LoraTEParallelLinear(
                base_layer=target, 
                adapter_name=adapter_name, 
                te_backend=megatron_core.extensions.transformer_engine, 
                backend=megatron_core.tensor_parallel, 
                **megatron_kwargs
            )

    

    return new_module
