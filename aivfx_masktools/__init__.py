"""AIVFX mask tools — batch-aware mask post-processing for the AI-VFX pipeline.

MaskAreaFilter: per-frame connected-component cleanup for a MASK batch.
  - area filter: drops white blobs whose pixel area is below `min_area`
    (removes small background detections / pedestrians)
  - fill_holes: closes internal holes inside the kept blobs
    (removes SAM3 segmentation holes that become grey patches in the
     driving plate -> black artifacts on the subject)

Batch-aware (every frame processed, frame count preserved). Pure downstream
mask op — no model, no effect on how frames are generated. With `enabled` off
it passes the mask through unchanged.
"""

import logging

import numpy as np
import torch

try:
    import scipy.ndimage as ndi
except Exception as ex:  # scipy ships with ComfyUI deps; fail loud if not
    ndi = None
    logging.error(f"[aivfx_masktools] scipy unavailable: {ex}")


class MaskAreaFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_area": ("INT", {"default": 3000, "min": 0, "max": 16777216, "step": 100}),
                "fill_holes": ("BOOLEAN", {"default": True, "label_on": "fill", "label_off": "off"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "filter", "label_off": "passthrough"}),
                "log_areas": ("BOOLEAN", {"default": True, "label_on": "log", "label_off": "quiet"}),
            },
            "optional": {
                # keep_largest > 0: per frame, keep ONLY the N biggest blobs (the main subject(s)),
                # dropping everyone else. Robust "main character" rule for a motion-robust but
                # ID-less semantic mask (SAM3): auto-adapts as the subject moves/scales, unlike a
                # fixed min_area. 0 = off. Applied AFTER min_area.
                "keep_largest": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "filter_area"
    CATEGORY = "AIVFX/mask"
    DESCRIPTION = (
        "Per-frame connected-component cleanup. Drops blobs below min_area (small "
        "background detections) and optionally fills internal holes inside kept "
        "blobs (SAM3 segmentation holes). Batch-aware; preserves frame count."
    )

    def filter_area(self, mask, min_area, fill_holes, enabled, log_areas, keep_largest=0):
        if not enabled or ndi is None:
            return (mask,)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B, H, W = mask.shape
        structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
        out = torch.zeros_like(mask)

        kept_total = 0
        dropped_total = 0
        for b in range(B):
            m_np = (mask[b].detach().cpu().numpy() > 0.5).astype(np.uint8)
            if m_np.sum() == 0:
                continue

            keep = m_np
            if min_area > 0 or keep_largest > 0:
                labeled, n = ndi.label(m_np, structure=structure)
                if n == 0:
                    continue
                areas = np.bincount(labeled.ravel())
                areas[0] = 0  # background label never kept
                passing = np.nonzero(areas >= max(min_area, 1))[0]  # drop tiny blobs first
                if keep_largest > 0 and passing.size > keep_largest:
                    # keep only the N biggest survivors (the main subject(s))
                    passing = passing[np.argsort(areas[passing])[::-1][:keep_largest]]
                keep = np.isin(labeled, passing).astype(np.uint8)
                kept_total += int(passing.size)
                dropped_total += int(n - passing.size)
                if log_areas:
                    comp_areas = sorted((int(a) for a in areas[1:]), reverse=True)
                    logging.info(f"[MaskAreaFilter] frame {b}: areas={comp_areas} min_area={min_area} keep_largest={keep_largest}")

            if fill_holes:
                keep = ndi.binary_fill_holes(keep, structure=structure).astype(np.uint8)

            out[b] = torch.from_numpy(keep.astype(np.float32)).to(mask.device)

        logging.info(
            f"[MaskAreaFilter] kept={kept_total} dropped={dropped_total} "
            f"over {B} frames (min_area={min_area}, keep_largest={keep_largest}, fill_holes={fill_holes})"
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"MaskAreaFilter": MaskAreaFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskAreaFilter": "Mask Area Filter (AIVFX)"}
