from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

try:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
except Exception:  # pragma: no cover
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class  # type: ignore

    QUANT_ALGOS = {}


"""
 This stuff is for systematically fully_shard from bottom up,
 with "*-1 parents are sharded, then continue up to root*", the def collect_bottom_up_shard_order is the function
 the tree looks like this:

model                          [FSDP]
├── block0                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   │   ├── q.weight           [SHARDED]
│   │   ├── q.scale            [IGNORED]
│   │   ├── k.weight           [SHARDED]
│   │   ├── k.scale            [IGNORED]
│   │   ├── v.weight           [SHARDED]
│   │   └── v.scale            [IGNORED]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   │   ├── weight             [SHARDED]
│   │   └── scale              [IGNORED]
│   └── conv                   [FSDP, ignored_params={} ]
│       ├── weight             [SHARDED]
│       └── bias               [SHARDED]
│
├── block1                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   └── conv                   [FSDP]
│
└── block2                     [FSDP]
    ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
    ├── ffn                    [FSDP, ignored_params={scale}]
    └── conv                   [FSDP]
"""


def freeze_and_detect_qt(model: torch.nn.Module) -> bool:
    has_qt = False
    for param in model.parameters():
        param.requires_grad = False
        local = getattr(param, "_local_tensor", None)
        if isinstance(param, QuantizedTensor) or isinstance(local, QuantizedTensor):
            has_qt = True
    return has_qt


def _mod_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _module_has_subtree_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=True))


def _module_has_direct_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=False))


def _children_with_params(name: str, module: torch.nn.Module, named_map: dict[str, torch.nn.Module]) -> list[str]:
    out: list[str] = []
    for child_name, _child in module.named_children():
        full = _mod_name(name, child_name)
        if _module_has_subtree_params(named_map[full]):
            out.append(full)
    return out


def _is_descendant(path: str, ancestor: str) -> bool:
    if ancestor == "":
        return path != ""
    return path == ancestor or path.startswith(ancestor + ".")


def _collect_leaf_parent_targets(model: torch.nn.Module) -> set[str]:
    named = dict(model.named_modules())
    all_names = [name for name in named.keys() if name != ""]

    structural_groups: set[str] = set()
    for name in all_names:
        children = _children_with_params(name, named[name], named)
        if len(children) >= 2:
            structural_groups.add(name)

    targets: set[str] = set(structural_groups)
    for name in all_names:
        if _module_has_direct_params(named[name]):
            if not any(_is_descendant(name, group_name) for group_name in structural_groups):
                targets.add(name)

    return targets


def _add_ancestors_to_root(targets: set[str]) -> set[str]:
    out = set(targets)
    for target in list(targets):
        cur = target
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            out.add(cur)
    out.add("")
    return out


def _depth(name: str) -> int:
    return 0 if name == "" else name.count(".") + 1


def _supports_fully_shard(module: torch.nn.Module) -> bool:
    return type(module).forward is not torch.nn.Module.forward


def collect_bottom_up_shard_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    named = dict(model.named_modules())
    leaf_parents = _collect_leaf_parent_targets(model)
    all_targets = _add_ancestors_to_root(leaf_parents)
    ordered_names = sorted(all_targets, key=_depth, reverse=True)

    out: list[tuple[str, torch.nn.Module]] = []
    for name in ordered_names:
        module = model if name == "" else named[name]
        if _supports_fully_shard(module):
            out.append((name, module))
    return out


def collect_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "scale" in param_name:
            ignored.add(param)
    return ignored


def collect_input_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "input_scale" in param_name:
            ignored.add(param)
    return ignored


def collect_scalar_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if param.ndim == 0:
            ignored.add(param)
    return ignored


def _has_odd_shard_dim(param: torch.Tensor) -> bool:
    return param.ndim > 0 and (int(param.shape[0]) % 2) == 1


def collect_odd_dim0_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if _has_odd_shard_dim(param):
            ignored.add(param)
    return ignored


def _should_materialize_unsharded_param(
    param_name: str,
    param: torch.Tensor,
    full_sd: dict[str, Any] | None = None,
) -> bool:
    if param_name.endswith("input_scale") or param.ndim == 0:
        return True
    if not _has_odd_shard_dim(param):
        return False
    if full_sd is not None and _is_quant_param(param_name, full_sd, param):
        return False
    return True


def _get_parent_module_and_name(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in param_name:
        return model, param_name
    parent_name, leaf_name = param_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _maybe_collapse_replicated_leading_dim(full_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    expected_shape = tuple(target_shape)
    if tuple(full_tensor.shape) == expected_shape:
        return full_tensor
    if full_tensor.ndim == 0 or full_tensor.ndim != len(expected_shape):
        return full_tensor
    if tuple(full_tensor.shape[1:]) != expected_shape[1:]:
        return full_tensor

    actual_leading = full_tensor.shape[0]
    expected_leading = expected_shape[0]
    if expected_leading <= 0 or actual_leading < expected_leading or actual_leading % expected_leading != 0:
        return full_tensor

    replicas = actual_leading // expected_leading
    if replicas <= 1:
        return full_tensor

    collapsed = full_tensor.reshape(expected_leading, replicas, *full_tensor.shape[1:])
    canonical = collapsed[:, 0, ...]
    if torch.equal(collapsed, canonical.unsqueeze(1).expand_as(collapsed)):
        return canonical
    return full_tensor


def _materialize_unsharded_param(
    model: torch.nn.Module,
    param_name: str,
    meta_param: torch.Tensor,
    full_tensor: torch.Tensor,
    device: torch.device,
    cpu_offload: bool,
) -> None:
    full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, meta_param.shape)
    full_tensor = full_tensor.to(dtype=meta_param.dtype, device=device)
    if cpu_offload:
        full_tensor = full_tensor.cpu()
    parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
    parent_module.register_parameter(
        leaf_name,
        torch.nn.Parameter(full_tensor, requires_grad=meta_param.requires_grad),
    )


def _materialize_missing_ignored_params(
    model: torch.nn.Module,
    full_sd: dict[str, Any],
    device: torch.device,
    strict: bool,
    cpu_offload: bool,
    release_sd: bool,
) -> None:
    for param_name, param in list(model.named_parameters()):
        if not getattr(param, "is_meta", False):
            continue
        if not _should_materialize_unsharded_param(param_name, param, full_sd):
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        _materialize_unsharded_param(model, param_name, param, full_tensor, device, cpu_offload)
        if release_sd:
            full_sd[param_name] = None


def _collect_subtree_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    return set(module.parameters())


def fully_shard_bottom_up(
    model: torch.nn.Module,
    fsdp_kwargs: dict[str, Any],
    native_ignore_scale: bool,
    ignored_modules: set[torch.nn.Module] | None = None,
) -> int:
    excluded_params: set[torch.nn.Parameter] = set()
    ignored_module_ids: set[int] = set()
    if ignored_modules:
        for mod in ignored_modules:
            excluded_params |= _collect_subtree_params(mod)
            ignored_module_ids.update(id(child) for child in mod.modules())

    shard_order = collect_bottom_up_shard_order(model)
    if ignored_modules:
        shard_order = [(n, m) for n, m in shard_order if id(m) not in ignored_module_ids]

    num_layers_sharded = 0
    for _name, module in shard_order:
        kwargs = dict(fsdp_kwargs)
        ignored_params: set[torch.nn.Parameter] = set()
        if native_ignore_scale:
            ignored_params |= collect_scale_ignored_params(module)

        ignored_params |= collect_input_scale_ignored_params(module)
        ignored_params |= collect_scalar_ignored_params(module)
        ignored_params |= collect_odd_dim0_ignored_params(module)

        subtree_params = set(module.parameters())
        ignored_params |= (excluded_params & subtree_params)

        if ignored_params:
            kwargs["ignored_params"] = ignored_params

        fully_shard(module, **kwargs)
        num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")
    return num_layers_sharded


def materialize_excluded_params(
    model: torch.nn.Module,
    excluded_modules: set[torch.nn.Module],
    full_sd: dict[str, Any],
    device: torch.device,
    cpu_offload: bool = False,
) -> int:
    """Load parameters of modules excluded from FSDP wrapping from a full state dict.

    After FSDP wrapping, excluded-module parameters remain on meta device.
    This function materializes them onto *device* (or CPU when cpu_offload).
    Returns the number of parameters materialized.
    """
    module_to_prefix: dict[int, str] = {}
    for name, mod in model.named_modules():
        module_to_prefix[id(mod)] = name

    count = 0
    for module in excluded_modules:
        prefix = module_to_prefix.get(id(module), "")
        for param_name, param in module.named_parameters(recurse=True):
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if not getattr(param, "is_meta", False):
                continue
            full_tensor = full_sd.get(full_name)
            if full_tensor is None:
                continue
            _materialize_unsharded_param(model, full_name, param, full_tensor, device, cpu_offload)
            count += 1
    return count


def _decode_comfy_quant(conf: Any) -> dict[str, Any] | None:
    if conf is None:
        return None
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, (bytes, bytearray)):
        return json.loads(conf.decode("utf-8"))
    if isinstance(conf, torch.Tensor):
        raw = conf.detach().cpu().numpy().tobytes()
        if conf.dtype == torch.uint8:
            return json.loads(raw)
        return json.loads(raw.decode("utf-8"))
    if isinstance(conf, str):
        return json.loads(conf)
    raise TypeError(f"Unsupported comfy_quant type: {type(conf)}")


def _find_scaled_fp8_key(full_sd: dict[str, Any]) -> str | None:
    if "scaled_fp8" in full_sd:
        return "scaled_fp8"

    for key in full_sd.keys():
        if key.endswith(".scaled_fp8"):
            return key

    return None


def _legacy_scaled_fp8_conf(prefix: str, full_sd: dict[str, Any]) -> dict[str, Any] | None:
    has_legacy_scale = f"{prefix}scale_weight" in full_sd
    has_converted_scale = f"{prefix}weight_scale" in full_sd
    if not has_legacy_scale and not has_converted_scale:
        return None

    conf: dict[str, Any] = {"format": "float8_e4m3fn"}
    scaled_fp8_key = _find_scaled_fp8_key(full_sd)
    if scaled_fp8_key is not None:
        scaled_fp8_weight = full_sd.get(scaled_fp8_key)
        if isinstance(scaled_fp8_weight, torch.Tensor) and scaled_fp8_weight.nelement() == 2:
            conf["full_precision_matrix_mult"] = True

    return conf


def _quant_payload_debug_info(param_name: str, full_sd: dict[str, Any]) -> str:
    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else param_name
    debug_bits = {
        "weight": param_name in full_sd,
        "comfy_quant": f"{prefix}comfy_quant" in full_sd,
        "weight_scale": f"{prefix}weight_scale" in full_sd,
        "weight_scale_2": f"{prefix}weight_scale_2" in full_sd,
        "input_scale": f"{prefix}input_scale" in full_sd,
        "legacy_scale_weight": f"{prefix}scale_weight" in full_sd,
        "legacy_scale_input": f"{prefix}scale_input" in full_sd,
        "scaled_fp8": _find_scaled_fp8_key(full_sd) is not None,
    }
    prefix_keys = sorted(
        key
        for key in full_sd.keys()
        if key.startswith(prefix)
        and (
            key == param_name
            or key.endswith("comfy_quant")
            or key.endswith("weight_scale")
            or key.endswith("weight_scale_2")
            or key.endswith("input_scale")
            or key.endswith("scale_weight")
            or key.endswith("scale_input")
        )
    )
    return f"payload={debug_bits}, prefix_keys={prefix_keys}"


def _shard_tensor(
    full_tensor: torch.Tensor,
    sharded_meta_param: Any,
    device: torch.device,
    *,
    pad_to_local_meta: bool = True,
) -> torch.Tensor:
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)

    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")

    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)).rank()

    chunk = torch.tensor_split(full_tensor, shard_world_size, dim=0)[shard_rank].to(device=device)

    local_meta = getattr(sharded_meta_param, "_local_tensor", None)
    if not pad_to_local_meta or not isinstance(local_meta, torch.Tensor):
        return chunk

    local_shape = tuple(local_meta.shape)
    if tuple(chunk.shape) == local_shape:
        return chunk
    if len(local_shape) != chunk.ndim:
        return chunk
    if any(local_dim < chunk_dim for local_dim, chunk_dim in zip(local_shape, chunk.shape)):
        return chunk

    sharded_param = full_tensor.new_zeros(local_shape, device=device)
    if chunk.numel() > 0:
        sharded_param[tuple(slice(0, dim) for dim in chunk.shape)].copy_(chunk)
    return sharded_param


def _is_quant_param(param_name: str, full_sd: dict[str, Any], sharded_meta_param: Any) -> bool:
    if isinstance(full_sd.get(param_name), QuantizedTensor):
        return True

    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else None
    if prefix is not None and (
        f"{prefix}comfy_quant" in full_sd
        or f"{prefix}weight_scale" in full_sd
        or f"{prefix}scale_weight" in full_sd
    ):
        return True

    if isinstance(sharded_meta_param, QuantizedTensor):
        return True
    if hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        return True

    return False


def _build_quantized_tensor(
    param_name: str,
    full_sd: dict[str, Any],
    sharded_meta_param: Any,
    device: torch.device,
):
    def _local_orig_shape(layout_name: str, local_qdata: torch.Tensor, logical_orig_shape: tuple[int, ...] | None) -> tuple[int, ...]:
        if logical_orig_shape is None:
            return tuple(local_qdata.shape)
        if layout_name == "TensorCoreNVFP4Layout" and len(logical_orig_shape) == 2 and local_qdata.dim() == 2:
            return (int(local_qdata.shape[0]), int(logical_orig_shape[1]))
        return tuple(local_qdata.shape)

    if not param_name.endswith("weight"):
        return None

    full_q = full_sd.get(param_name)
    if isinstance(full_q, QuantizedTensor):
        qt = cast(Any, full_q)
        local_qdata = _shard_tensor(qt._qdata.to(device=device), sharded_meta_param, device, pad_to_local_meta=False)
        local_params = replace(
            qt._params, orig_shape=_local_orig_shape(qt._layout_cls, local_qdata, getattr(qt._params, "orig_shape", None))
        )
        return QuantizedTensor(local_qdata, qt._layout_cls, local_params)

    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        conf = _legacy_scaled_fp8_conf(prefix, full_sd)
    if conf is None:
        return None

    quant_format = conf.get("format", None)
    if quant_format is None or quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unknown quantization format for {param_name}: {quant_format}")
    if quant_format == "mxfp8":
        raise NotImplementedError(
            "Raylight FSDP does not support MXFP8 quantized weights yet. "
            "Use FP8/NVFP4 weights or disable Raylight FSDP quant loading."
        )

    qconfig = QUANT_ALGOS[quant_format]
    layout_name = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise ValueError(f"Missing layout class for {layout_name}")

    full_qdata = full_sd.get(param_name)
    if full_qdata is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    qdata = full_qdata.to(device=device, dtype=qconfig["storage_t"])
    qdata = _shard_tensor(qdata, sharded_meta_param, device, pad_to_local_meta=False)

    params_kwargs: dict[str, Any] = {"orig_shape": tuple(qdata.shape)}

    local_meta = None
    if isinstance(sharded_meta_param, QuantizedTensor):
        local_meta = sharded_meta_param
    elif hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        local_meta = sharded_meta_param._local_tensor
    orig_dtype = None
    if local_meta is not None and hasattr(local_meta, "_params"):
        orig_dtype = getattr(local_meta._params, "orig_dtype", None)
    if orig_dtype is None:
        orig_dtype = getattr(sharded_meta_param, "dtype", None)
    if orig_dtype is not None:
        params_kwargs["orig_dtype"] = orig_dtype

    logical_orig_shape = getattr(getattr(local_meta, "_params", None), "orig_shape", None)
    if logical_orig_shape is None and quant_format == "nvfp4" and full_qdata.dim() == 2:
        logical_orig_shape = (int(full_qdata.shape[0]), int(full_qdata.shape[1] * 2))
    params_kwargs["orig_shape"] = _local_orig_shape(layout_name, qdata, logical_orig_shape)

    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scale = full_sd.get(f"{prefix}weight_scale")
        if scale is None:
            scale = full_sd.get(f"{prefix}scale_weight")
        if scale is not None:
            scale = scale.to(device=device)
        params_kwargs["scale"] = scale
    elif quant_format == "nvfp4":
        tensor_scale = full_sd.get(f"{prefix}weight_scale_2")
        block_scale = full_sd.get(f"{prefix}weight_scale")
        if tensor_scale is None or block_scale is None:
            raise ValueError(f"Missing NVFP4 scales for {param_name}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.view(dtype=torch.float8_e4m3fn).to(device=device)
        block_scale = _shard_tensor(block_scale, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["scale"] = tensor_scale
        params_kwargs["block_scale"] = block_scale
    elif quant_format == "gguf":
        n_blocks_per_superblock = conf.get("n_blocks_per_superblock", 8)
        super_block_scale_scale = full_sd.get(f"{prefix}super_block_scale_scale")
        super_block_min_scale = full_sd.get(f"{prefix}super_block_min_scale")
        quantized_block_scale = full_sd.get(f"{prefix}quantized_block_scale")
        quantized_block_min = full_sd.get(f"{prefix}quantized_block_min")
        if (
            super_block_scale_scale is None
            or super_block_min_scale is None
            or quantized_block_scale is None
            or quantized_block_min is None
        ):
            raise ValueError(f"Missing GGUF scales for {param_name}")
        super_block_scale_scale = super_block_scale_scale.to(device=device)
        super_block_min_scale = super_block_min_scale.to(device=device)
        quantized_block_scale = quantized_block_scale.to(device=device)
        quantized_block_min = quantized_block_min.to(device=device)
        super_block_scale_scale = _shard_tensor(super_block_scale_scale, sharded_meta_param, device, pad_to_local_meta=False)
        super_block_min_scale = _shard_tensor(super_block_min_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_scale = _shard_tensor(quantized_block_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_min = _shard_tensor(quantized_block_min, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["n_blocks_per_superblock"] = n_blocks_per_superblock
        params_kwargs["super_block_scale_scale"] = super_block_scale_scale
        params_kwargs["super_block_min_scale"] = super_block_min_scale
        params_kwargs["quantized_block_scale"] = quantized_block_scale
        params_kwargs["quantized_block_min"] = quantized_block_min
        if f"{prefix}scale" in full_sd:
            scale = full_sd.get(f"{prefix}scale")
            if scale is not None:
                params_kwargs["scale"] = scale.to(device=device)
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    params = layout_cls.Params(**params_kwargs)
    return QuantizedTensor(qdata, layout_name, params)


def _release_quant_keys(full_sd: dict[str, Any], param_name: str) -> None:
    prefix = param_name[: -len("weight")]
    for key in (
        param_name,
        f"{prefix}weight_scale",
        f"{prefix}weight_scale_2",
        f"{prefix}input_scale",
        f"{prefix}scale_weight",
        f"{prefix}scale_input",
        f"{prefix}comfy_quant",
        f"{prefix}super_block_scale_scale",
        f"{prefix}super_block_min_scale",
        f"{prefix}quantized_block_scale",
        f"{prefix}quantized_block_min",
    ):
        if key in full_sd:
            full_sd[key] = None


# Heavily modified from
# https://github.com/meta-pytorch/torchtune/blob/d0f63bb33d00b8bd3905a010b71d8c6324c2e980/torchtune/training/_distributed.py#L336
# Need to be done since dcp loader cause wrong dtype among rank when broadcasting.
def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict=False,
    cpu_offload=False,
    release_sd=True,
):
    meta_sharded_sd = model.state_dict()
    sharded_sd: dict[str, torch.nn.Parameter] = {}
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if _should_materialize_unsharded_param(param_name, sharded_meta_param, full_sd):
            full_tensor = full_sd.get(param_name)
            if full_tensor is None:
                if strict:
                    raise ValueError(f"Missing parameter {param_name} in state_dict")
                continue
            _materialize_unsharded_param(model, param_name, sharded_meta_param, full_tensor, device, cpu_offload)
            if release_sd:
                full_sd[param_name] = None
            continue

        if _is_quant_param(param_name, full_sd, sharded_meta_param):
            quant_tensor = _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device)
            if quant_tensor is None:
                raise ValueError(
                    f"Expected quantized tensor for {param_name}, but could not build it ({_quant_payload_debug_info(param_name, full_sd)})"
                )
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = DTensor.from_local(
                    quant_tensor,
                    device_mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            if release_sd:
                _release_quant_keys(full_sd, param_name)
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        if not hasattr(sharded_meta_param, "device_mesh"):
            full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, sharded_meta_param.shape)
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if hasattr(sharded_meta_param, "device_mesh"):
            local_dense = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                local_dense,
                device_mesh=sharded_meta_param.device_mesh,
                placements=sharded_meta_param.placements,
            )
        else:
            sharded_tensor = full_tensor
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None
    out = model.load_state_dict(sharded_sd, strict=strict, assign=True)
    _materialize_missing_ignored_params(model, full_sd, device, strict, cpu_offload, release_sd)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PRESHARD CACHE (b2) — appended by boot patch. Persist per-rank FSDP shards to
# disk so a fresh worker skips the ~130s per-tensor fp8 scatter. Deps (QuantizedTensor,
# get_layout_class, DTensor, torch) already imported above.
# ═══════════════════════════════════════════════════════════════════════════
import os as _ps_os
import time as _ps_time
import hashlib as _ps_hashlib


def _extract_local(param):
    data = param.data if isinstance(param, torch.nn.Parameter) else param
    if isinstance(data, DTensor):
        return data._local_tensor
    return data


def _serialize_local(t):
    if hasattr(t, '_qdata') and hasattr(t, '_layout_cls') and hasattr(t, '_params'):
        layout = t._layout_cls
        if isinstance(layout, str):
            ln = layout
        elif hasattr(layout, '__name__'):
            ln = layout.__name__
        else:
            ln = type(layout).__name__
        return {'q': True, 'qd': t._qdata.detach().cpu(), 'ln': ln, 'p': t._params}
    return {'q': False, 'd': t.detach().cpu()}


def _deserialize_local(entry, device):
    if entry['q']:
        cls = get_layout_class(entry['ln'])
        p = entry['p']
        # Move Params tensor fields (scale, block_scale, ...) onto the device so nothing stays a CPU
        # view onto the mmap (matches the full-scatter path; avoids a render-time page fault that on a
        # network FUSE file could SIGBUS). Defensive: any issue -> pass Params through unchanged.
        try:
            import dataclasses as _dc
            if _dc.is_dataclass(p):
                _upd = {f.name: getattr(p, f.name).to(device)
                        for f in _dc.fields(p)
                        if torch.is_tensor(getattr(p, f.name, None))}
                if _upd:
                    p = _dc.replace(p, **_upd)
        except Exception:
            p = entry['p']
        return QuantizedTensor(entry['qd'].to(device), cls, p)
    return entry['d'].to(device)


def _verify_bytes(path, saved, rank, nthreads=8):
    """CONTROL for the mmap load: re-read the SAME file the buffered way (_parallel_read) and
    byte-compare a deterministic 16-key sample of loaded entries against the mmap-loaded ones.
    Proves mmap-mapped bytes == buffered-read bytes (silent-corruption discriminator) + checks the
    mmap data is finite. In-process, same boot, no extra GPU render. Non-fatal; returns PASS/FAIL bool."""
    try:
        _bio2 = _parallel_read(path, nthreads)
        saved2 = torch.load(_bio2, map_location='cpu', weights_only=False)
        _bio2 = None
    except Exception as _e:
        print(f"[Rank {rank}] PRESHARD_VERIFY skipped (control load failed: {_e})", flush=True)
        return True

    def _eq(a, b):
        try:
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                return a == b
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
        except Exception:
            try:
                return bool(torch.equal(a.float(), b.float()))
            except Exception:
                return False

    keys = sorted(k for k in saved.keys() if k != '__meta__')
    step = max(1, len(keys) // 16)
    sample = keys[::step] if keys else []
    n = qd_ok = sc_ok = fin_ok = 0
    bad = []
    for k in sample:
        e1 = saved.get(k); e2 = saved2.get(k)
        if e1 is None or e2 is None:
            continue
        n += 1
        a = e1.get('qd', e1.get('d')); b = e2.get('qd', e2.get('d'))
        if _eq(a, b):
            qd_ok += 1
        else:
            bad.append(k)
        try:
            fin_ok += 1 if bool(torch.isfinite(a.float()).all()) else 0
        except Exception:
            fin_ok += 1
        s1 = getattr(e1.get('p'), 'scale', None); s2 = getattr(e2.get('p'), 'scale', None)
        if torch.is_tensor(s1) and torch.is_tensor(s2):
            sc_ok += 1 if _eq(s1, s2) else 0
        else:
            sc_ok += 1
    saved2 = None
    ok = (n > 0 and qd_ok == n and fin_ok == n and sc_ok == n)
    line = (f"[Rank {rank}] PRESHARD_VERIFY sample={n} qd_match={qd_ok}/{n} scale_match={sc_ok}/{n} "
            f"finite={fin_ok}/{n} -> {'PASS' if ok else 'FAIL'}"
            + (f" MISMATCH={bad[:5]}" if bad else ""))
    print(line, flush=True)
    try:
        with open("/runpod-volume/preshard_load_log.txt", "a") as _lf:
            _lf.write(line + "\n")
    except Exception:
        pass
    return ok


def preshard_version(full_sd, world_size):
    """Content-hash key: world_size + full structure + a value fingerprint. ANY change to
    model / LoRA / fp8 recipe / gpuCount ⇒ different key ⇒ cache miss ⇒ safe full scatter.
    Never returns a colliding key for a different config; on error raises (caller scatters)."""
    h = _ps_hashlib.sha1()
    h.update(str(int(world_size)).encode())
    items = sorted(full_sd.items(), key=lambda kv: kv[0])
    for name, v in items:
        t = getattr(v, '_qdata', None)
        if t is None:
            t = v.data if hasattr(v, 'data') else v
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
    # value fingerprint: ~16 evenly-spaced tensors → catches value-only changes (LoRA merge)
    n = len(items)
    if n:
        step = max(1, n // 16)
        for i in range(0, n, step):
            try:
                name, v = items[i]
                t = getattr(v, '_qdata', None)
                if t is None:
                    t = v.data if hasattr(v, 'data') else v
                flat = t.detach().reshape(-1)
                s = max(1, flat.numel() // 64)
                h.update(repr(float(flat[::s].float().sum().item())).encode())
            except Exception:
                pass
    return h.hexdigest()[:16]


def save_fsdp_shards(diffusion_model, path, rank, world_size):
    """Save per-rank FSDP shard tensors after the full scatter + materialize. Atomic write."""
    t0 = _ps_time.time()
    shard = {}
    for name, param in diffusion_model.named_parameters():
        shard[name] = _serialize_local(_extract_local(param))
    shard['__meta__'] = {'ws': world_size, 'r': rank, 'n': len(shard) - 1}
    _ps_os.makedirs(_ps_os.path.dirname(path), exist_ok=True)
    # Direct save to final — os.replace(rename) proved UNRELIABLE on this MooseFS volume (8GB tmp
    # landed but rename didn't). Non-atomic, but safe: load() has try/except fallback to full scatter,
    # so a partial/corrupt file just scatters (never wrong weights). Writes are idempotent per key.
    torch.save(shard, path)
    sz = _ps_os.path.getsize(path) / (1024 * 1024)
    print(f"[Rank {rank}] Saved {len(shard)-1} pre-shard params → {path} ({sz:.0f} MB, {_ps_time.time()-t0:.1f}s)", flush=True)

    # LRU: keep the newest FSDP_SHARD_KEEP config-version dirs, prune older (rank 0 only, non-fatal).
    # Each version is ~ws×8.5GB, so unbounded configs WILL fill the volume (it bit us 7/05).
    if rank == 0:
        try:
            import shutil as _sh
            keep = int(_ps_os.environ.get("FSDP_SHARD_KEEP", "3"))
            root = _ps_os.path.dirname(_ps_os.path.dirname(path))  # = FSDP_SHARD_DIR
            vers = [(_ps_os.path.getmtime(_ps_os.path.join(root, d)), d)
                    for d in _ps_os.listdir(root)
                    if _ps_os.path.isdir(_ps_os.path.join(root, d))]
            vers.sort(reverse=True)  # newest first
            for _mt, d in vers[keep:]:
                _sh.rmtree(_ps_os.path.join(root, d), ignore_errors=True)
                print(f"[Rank {rank}] LRU pruned old shard version {d}", flush=True)
        except Exception as _e:
            print(f"[Rank {rank}] LRU prune failed (non-fatal): {_e}", flush=True)


def _parallel_read(path, nthreads=8):
    """Read a file into memory with N parallel streams — MooseFS stripes across chunkservers,
    so parallel reads are ~2.4x+ faster than single-stream (measured 319→777 MB/s @4 streams).
    Returns a BytesIO for torch.load. Needs ~filesize RAM (box has 500GB)."""
    import io as _io, threading as _th
    global _LAST_READ_TIMES, _LAST_READ_STEPS
    sz = _ps_os.path.getsize(path)
    _ta = _ps_time.time()
    buf = bytearray(sz)                    # STEP 1: allocate + zero-fill an 8GB buffer
    _t_alloc = _ps_time.time() - _ta
    n = max(1, nthreads)
    chunk = (sz + n - 1) // n
    _LAST_READ_TIMES = [None] * n          # per-thread [elapsed_s, MBps] -> is one thread a straggler?

    def _rd(i):
        off = i * chunk
        end = min(off + chunk, sz)
        if off >= end:
            return
        _s = _ps_time.time(); _n = 0
        _start_off = _s - _tr              # when this thread ACTUALLY began, vs spawn -> exposes GIL stagger
        _t_open0 = _ps_time.time()
        mv = memoryview(buf)[off:end]
        f = open(path, 'rb', buffering=0); f.seek(off)
        _t_open = _ps_time.time() - _t_open0
        _t_io0 = _ps_time.time(); got = 0
        while got < (end - off):
            r = f.readinto(mv[got:])
            if not r:
                break
            got += r; _n += r
        _t_io = _ps_time.time() - _t_io0
        f.close()
        _dt = _ps_time.time() - _s
        _LAST_READ_TIMES[i] = {"start_off": round(_start_off, 2), "open_seek": round(_t_open, 3),
                               "readinto": round(_t_io, 2), "total": round(_dt, 2),
                               "MBps": round(_n / 2**20 / _t_io) if _t_io else 0}

    _tr = _ps_time.time()
    ths = [_th.Thread(target=_rd, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    _t_readwall = _ps_time.time() - _tr    # STEP 2: wall time of the 8 read threads (start->join)
    _tb = _ps_time.time()
    out = _io.BytesIO(buf)                 # STEP 3: copy the whole 8GB buffer into a BytesIO
    _LAST_READ_STEPS = {"alloc_s": round(_t_alloc, 2), "read_wall_s": round(_t_readwall, 2),
                        "bytesio_copy_s": round(_ps_time.time() - _tb, 2)}
    return out


def load_fsdp_shards(diffusion_model, path, device):
    """Load pre-sharded tensors into an FSDP-wrapped model (meta params carry
    device_mesh/placements). Replaces the full scatter + materialize."""
    t0 = _ps_time.time()
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    ws = dist.get_world_size() if dist.is_initialized() else 1

    # HOST-STORE REMAP (2026-07-10): HOSTSTORE_SHARDS_DIR is set by the handler only when RunPod
    # staged the repo's fsdp_shards_pr/ onto host NVMe (12.4 GB/s measured vs ~0.4 GB/s volume).
    # Load remaps to the staged copy when the exact <hash>/<rank file> exists there; saves and
    # every miss keep the volume path untouched.
    _hsd = _ps_os.environ.get("HOSTSTORE_SHARDS_DIR", "")
    if _hsd:
        _cand = _ps_os.path.join(_hsd, _ps_os.path.basename(_ps_os.path.dirname(path)),
                                 _ps_os.path.basename(path))
        if _ps_os.path.isfile(_cand):
            print(f"[b2][rank{rank}] shard load remapped to host store: {_cand}", flush=True)
            path = _cand
        else:
            print(f"[b2][rank{rank}] host store set but no staged copy of {_cand} — volume path kept", flush=True)

    # PHASE 1: get the shard dict into memory.
    #   mmap path (default): torch.load(path, mmap=True) maps the file so the OS serves bytes straight
    #   from page cache into torch storages — NO intermediate Python buffer. Kills the old 8GB bytearray
    #   zero-fill + 8GB BytesIO copy (measured ~10-14s of the 12-16s "parallel_read"); the byte movement
    #   folds into the .to(device) copy in PHASE 2 instead. FSDP_SHARD_MMAP=0 reverts to the 8-stream
    #   read (kept as an A/B lever + fallback for a genuinely COLD MooseFS read prewarm didn't cover).
    _nrd = int(_ps_os.environ.get("FSDP_SHARD_READ_THREADS", "8"))
    # HYBRID (2026-07-09, same-host eviction A/B, worker wmi26ane9dsqmq): mmap wins CACHED by 22x
    # (1.0s vs 22.5s) but loses COLD-over-network 1.3-5x (44-165s vs 33-43s: serial page faults pull
    # as little as 50 MB/s from MooseFS where 8 explicit streams hold 191-253 MB/s; confirmed in prod
    # same boot — rank0 mmap cold 38.9s vs rank1 warm 2.9s). FSDP_SHARD_MMAP: "1"=force mmap,
    # "0"=force 8-stream, "auto" (default)=mincore residency picks — >=50% resident -> mmap.
    def _resident_frac(_p):
        # mirrors the handler's production-proven _mincore_pct: ACCESS_COPY gives a WRITABLE (COW)
        # mapping — ctypes.from_buffer REQUIRES writable; a PROT_READ map raises TypeError (that bug
        # shipped 7/09 as silent resident_frac=-1.0 on both ranks; caught same day by the log line).
        _m = None
        try:
            import mmap as _mmod, ctypes as _ct
            _sz = _ps_os.path.getsize(_p)
            if not _sz: return 1.0
            _fd = _ps_os.open(_p, _ps_os.O_RDONLY)
            try:
                _m = _mmod.mmap(_fd, _sz, access=_mmod.ACCESS_COPY)
            finally:
                _ps_os.close(_fd)
            _pg = _ps_os.sysconf("SC_PAGE_SIZE")
            _npages = (_sz + _pg - 1) // _pg
            _vec = (_ct.c_ubyte * _npages)()
            _libc = _ct.CDLL("libc.so.6", use_errno=True)
            _buf = (_ct.c_char * _sz).from_buffer(_m)
            _rc = _libc.mincore(_ct.c_void_p(_ct.addressof(_buf)), _ct.c_size_t(_sz), _vec)
            _buf = None                                   # release ctypes export before mmap close
            if _rc != 0: return -2.0                      # errno path, distinct from exception path
            _step = max(1, (1024 * 1024) // _pg)          # sample one page per MiB
            _idx = range(0, _npages, _step)
            return sum(_vec[_i] & 1 for _i in _idx) / max(1, len(_idx))
        except Exception:
            return -1.0
        finally:
            try:
                if _m is not None: _m.close()
            except Exception: pass
    _mode_env = _ps_os.environ.get("FSDP_SHARD_MMAP", "auto")
    if _mode_env == "1":
        _use_mmap, _resfrac = True, None
    elif _mode_env == "0":
        _use_mmap, _resfrac = False, None
    else:
        _resfrac = _resident_frac(path)
        _use_mmap = _resfrac >= 0.5 or _resfrac < 0     # mincore failure -> mmap (the prior default)
    print(f"[b2][rank{rank}] loader mode={'mmap' if _use_mmap else '8stream'} "
          f"(env={_mode_env} resident_frac={None if _resfrac is None else round(_resfrac, 3)})", flush=True)
    def _cached_gib():
        try:
            for _l in open("/proc/meminfo"):
                if _l.startswith("Cached:"): return int(_l.split()[1]) / (1024 * 1024)
        except Exception: pass
        return 0.0
    _c0 = _cached_gib()
    _rt = None
    if _use_mmap:
        _tr = _ps_time.time()
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        _t_pread = 0.0                                # no separate read phase — storages are lazy mmap views
        _cdelta = _cached_gib() - _c0
        _t_read = _ps_time.time() - t0
        _stp = {"mode": "mmap", "torch_load_s": round(_ps_time.time() - _tr, 2)}
    else:
        _tr = _ps_time.time()
        _bio = _parallel_read(path, _nrd)
        _t_pread = _ps_time.time() - _tr
        _cdelta = _cached_gib() - _c0   # +~filesize GiB => read was COLD (faulted in); ~0 => cached (WARM)
        saved = torch.load(_bio, map_location='cpu', weights_only=False)
        _t_read = _ps_time.time() - t0
        _bio = None
        try: _stp = _LAST_READ_STEPS
        except Exception: _stp = None
        try: _rt = _LAST_READ_TIMES
        except Exception: _rt = None
    meta = saved.pop('__meta__')
    if meta['ws'] != ws:
        raise ValueError(f"Pre-shard world_size {meta['ws']} != current {ws}")
    if meta['r'] != rank:
        raise ValueError(f"Pre-shard rank {meta['r']} != current {rank}")

    if (_ps_os.environ.get("FSDP_SHARD_VERIFY", "0") == "1"
            or _ps_os.path.exists("/runpod-volume/code_hotpatch/.fsdp_verify")):
        _verify_bytes(path, saved, rank, _nrd)   # byte-identity control vs buffered read (no extra render)

    # PHASE 2: deserialize + CPU→GPU copy + DTensor reconstruct per param
    _t1 = _ps_time.time()
    meta_sd = diffusion_model.state_dict()
    sharded_sd = {}
    for name, meta_param in meta_sd.items():
        entry = saved.get(name)
        if entry is None:
            continue
        local = _deserialize_local(entry, device)
        if hasattr(meta_param, 'device_mesh'):
            sharded = DTensor.from_local(local, meta_param.device_mesh, meta_param.placements)
        else:
            sharded = local
        sharded_sd[name] = torch.nn.Parameter(sharded)
    _t_recon = _ps_time.time() - _t1

    # PHASE 3: apply into the model
    _t2 = _ps_time.time()
    diffusion_model.load_state_dict(sharded_sd, strict=False, assign=True)
    _t_apply = _ps_time.time() - _t2
    _mode = "mmap" if _use_mmap else f"read{_nrd}"
    _bd = (f"[Rank {rank}] PRESHARD_LOAD_BREAKDOWN mode={_mode} parallel_read={_t_pread:.1f}s cache_delta={_cdelta:.1f}GiB "
           f"unpickle={_t_read-_t_pread:.1f}s recon+gpucopy={_t_recon:.1f}s apply={_t_apply:.1f}s "
           f"total={_ps_time.time()-t0:.1f}s read_steps={_stp} read_threads[s,MBps]={_rt}")
    print(_bd, flush=True)
    try:  # reliable off-pod readback (stdout/comfy.log S3 view lag-truncates)
        with open("/runpod-volume/preshard_load_log.txt", "a") as _lf:
            _lf.write(_bd + "\n")
    except Exception:
        pass
    print(f"[Rank {rank}] Loaded {len(sharded_sd)} params from pre-shards ({_ps_time.time()-t0:.1f}s)", flush=True)
    try:  # PREWARM MANIFEST: record the exact file this rank loaded so the next boot's shard
          # prewarm warms the REAL bytes into host page cache (not stale versions). Fail-safe.
        _mdir = _ps_os.path.dirname(_ps_os.path.dirname(path))  # = FSDP_SHARD_DIR
        with open(_ps_os.path.join(_mdir, f".prewarm_manifest_rank{rank}"), "w") as _pf:
            _pf.write(path)
    except Exception:
        pass
