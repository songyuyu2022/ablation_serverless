# shared.py
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, Union, Optional

import torch


def tensor_to_b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_tensor(s: str, map_location: str = "cpu") -> torch.Tensor:
    raw = base64.b64decode(s.encode("utf-8"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location=map_location)


# ============================================================
# Existing API: pack/unpack (keep compatible)
# ============================================================

def pack(payload: Dict[str, Any], *, map_tensors: bool = True) -> Dict[str, Any]:
    """
    Convert any torch.Tensor in dict to base64 strings.
    """
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            out[k] = {"__tensor__": True, "b64": tensor_to_b64(v)}
        else:
            out[k] = v
    return out


def unpack(payload: Dict[str, Any], *, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Convert {"__tensor__":True,"b64":...} back to torch.Tensor.
    """
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, dict) and v.get("__tensor__") is True and "b64" in v:
            out[k] = b64_to_tensor(v["b64"], map_location=map_location)
        else:
            out[k] = v
    return out


# ============================================================
# Added API: tensor_to_pack / pack_to_tensor
# (for controller/pre_fn/expert_app/post_fn unified protocol)
# ============================================================

def tensor_to_pack(t: torch.Tensor) -> Dict[str, Any]:
    """
    Pack a single torch.Tensor into a JSON-serializable dict.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"tensor_to_pack expects torch.Tensor, got {type(t)}")
    return {"__tensor__": True, "b64": tensor_to_b64(t)}


def pack_to_tensor(obj: Any, *, map_location: str = "cpu") -> torch.Tensor:
    """
    Unpack a packed tensor dict back to torch.Tensor.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if not (isinstance(obj, dict) and obj.get("__tensor__") is True and "b64" in obj):
        raise TypeError(f"pack_to_tensor expects packed tensor dict, got {type(obj)}")
    return b64_to_tensor(obj["b64"], map_location=map_location)


# ============================================================
# Added API: dumps / loads for HTTP bodies
# ============================================================

def dumps(payload: Dict[str, Any]) -> bytes:
    """
    pack(payload) then JSON serialize to bytes for HTTP request body.
    """
    packed = pack(payload)
    # ensure_ascii=False so Chinese paths/names won't break; bytes for requests/aiohttp
    return json.dumps(packed, ensure_ascii=False).encode("utf-8")


def loads(raw: Union[bytes, str]) -> Dict[str, Any]:
    """
    Parse JSON (bytes/str) then unpack tensors back to torch.Tensor.
    """
    if isinstance(raw, (bytes, bytearray)):
        s = raw.decode("utf-8")
    elif isinstance(raw, str):
        s = raw
    else:
        raise TypeError(f"loads expects bytes or str, got {type(raw)}")

    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise TypeError(f"loads expects JSON object (dict), got {type(obj)}")
    return unpack(obj)
