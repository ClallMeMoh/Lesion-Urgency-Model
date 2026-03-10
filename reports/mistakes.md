# Mistakes Log

This file tracks bugs, wrong assumptions, and pipeline failures encountered
during development. Each entry includes what happened, the root cause, the fix
applied, and a prevention step (test, assertion, validation, or guardrail) to
avoid recurrence.

---

## Mistake 1: Invalid build backend in pyproject.toml

**What happened:** `pip install -e .` failed with `ModuleNotFoundError: No module named 'setuptools.backends'`.

**Root cause:** Used `setuptools.backends.legacy:build` as the build backend, which requires setuptools 75+
with the new backends module. The installed setuptools (75.1.0) does not expose this path as the
`setuptools.backends` module was never added to that version.

**Fix:** Changed build backend to `setuptools.build_meta`, which is the standard, widely-supported backend.

**Prevention:** Always use `setuptools.build_meta` as the default backend. The `setuptools.backends` path
is non-standard and should be avoided. Added assertion in CI: `pip install -e . && python -c "import urgency"`.

---

## Mistake 2: OmegaConf does not support `typing.Literal` type annotations

**What happened:** `OmegaConf.structured(AppConfig)` raised `ValidationError: Unexpected type annotation: Literal[uncertain, exclude]`.

**Root cause:** OmegaConf 2.3 does not support `typing.Literal` as a field type in structured configs
(dataclasses). Literal types are not part of OmegaConf's type system.

**Fix:** Replaced all `Literal[...]` type annotations in config dataclasses with plain `str`.
Runtime validation of allowed values is handled by the consuming code (splits.py, losses.py, trainer.py).

**Prevention:** When using OmegaConf structured configs, restrict field types to those explicitly
supported: `int`, `float`, `bool`, `str`, `list`, `dict`, and nested dataclasses. Add a comment
in `config.py` documenting valid values for string enum fields.

---

## Mistake 3: `OmegaConf.to_object()` returns dict instead of dataclass when loading plain YAML

**What happened:** `UrgencyInferer._load_config()` returned a `dict` instead of `AppConfig`,
causing `AttributeError: 'dict' object has no attribute 'model'`.

**Root cause:** `OmegaConf.load()` on a plain YAML file returns an unstructured `DictConfig`.
Calling `to_object()` on an unstructured DictConfig returns a `dict`, not a dataclass instance.

**Fix:** Added `load_app_config(path)` utility in `utils/io.py` that merges the loaded YAML with
`OmegaConf.structured(AppConfig)` before calling `to_object()`. The merge enforces the structured
schema and ensures the result is a typed dataclass. Added an `assert isinstance(result, AppConfig)`
guard to catch future regressions.

**Prevention:** Always use `load_app_config()` when reading a saved config YAML. Never call
`OmegaConf.to_object()` directly on a file-loaded DictConfig. The `load_app_config` utility is
the single authorized code path for this operation.

---

## Mistake 4: PyTorch 2.6 `weights_only=True` default blocks loading checkpoints with custom objects

**What happened:** `UrgencyLightningModule.load_from_checkpoint()` raised `UnpicklingError: Weights only load failed` because the checkpoint stored `AppConfig` as a hyperparameter (via `save_hyperparameters()`).

**Root cause:** PyTorch 2.6 changed the default `weights_only` argument from `False` to `True`.
Lightning's `load_from_checkpoint` stores hyperparameters (including `cfg: AppConfig`) in the checkpoint
using pickle, which is blocked by the new default.

**Fix:** Replaced `load_from_checkpoint` with a direct state_dict extraction approach:
1. Load checkpoint raw with `weights_only=False` (the checkpoint is local, not from an external source).
2. Extract `state_dict` key and strip the `"model."` prefix added by LightningModule.
3. Call `model.load_state_dict(model_state)` directly.

Added `load_model_weights(model, ckpt_path, device)` helper in `models/backbones.py`.

**Prevention:** Always load model weights via `load_model_weights()` helper, not `load_from_checkpoint`.
The config is always loaded separately from `config.yaml`, so embedding it in the checkpoint is
unnecessary. `UrgencyLightningModule` keeps `save_hyperparameters(ignore=["model"])` but this
still stores `cfg`, which is now ignored at load time.

---

## Mistake 5: Balanced subset lost patient_id column during sampling

**What happened:** After creating balanced subset, patient-level splits only produced 507 samples
(all minority classes, zero benign) instead of the expected ~3,118.

**Root cause:** Using `groupby().apply(func, include_groups=False)` dropped the `patient_id`
column from the sampled benign data. Without patient_id, patient-level splitting couldn't
group benign samples, so they were all excluded.

**Fix:** Removed `include_groups=False` parameter from the groupby call, keeping patient_id
in the returned DataFrame.

**Prevention:** After any data transformation step, verify the output DataFrame has all
expected columns and the expected number of rows before proceeding. Add shape assertions.

---

## Mistake 6: CPU-only PyTorch installed despite having NVIDIA GPU

**What happened:** Training was running on CPU despite RTX 4070 Laptop GPU being available.
`torch.cuda.is_available()` returned False.

**Root cause:** PyTorch was installed as `2.7.1+cpu` (CPU-only build) because the default pip
install doesn't include CUDA support. Must use the `--index-url` flag with CUDA-specific index.

**Fix:** Installed CUDA build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

**Prevention:** After installing PyTorch, always verify with `python -c "import torch; print(torch.cuda.is_available())"`.
Include the CUDA index URL in install instructions.

---

## Mistake 7: NumPy 2.x incompatibility with compiled packages

**What happened:** After installing CUDA PyTorch, scipy/matplotlib/torchmetrics crashed with
`_ARRAY_API not found` and similar NumPy ABI errors.

**Root cause:** New torch pulled numpy 2.3.5 but existing scipy, matplotlib, torchmetrics were
compiled against numpy 1.x ABI. The ABI changed between numpy 1.x and 2.x.

**Fix:** Upgraded all affected packages: scipy, matplotlib, torchmetrics, lightning, scikit-learn,
pandas. NumPy settled at 1.26.4 (compatible with all packages).

**Prevention:** When upgrading PyTorch, also upgrade the full scientific stack. Pin numpy to a
compatible version range in pyproject.toml.

---
