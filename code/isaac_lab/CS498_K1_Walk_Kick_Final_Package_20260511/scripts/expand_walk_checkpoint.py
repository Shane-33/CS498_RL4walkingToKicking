#!/usr/bin/env python3
"""Expand RSL-RL MLP checkpoints when observation size grows (e.g. walk 69 → kick 86).

Loads a `.pt` file created by Isaac Lab ``OnPolicyRunner`` (contains ``model_state_dict``).
For each 2-D weight tensor ``W`` with ``W.shape[1] == old_obs_dim`` and whose name matches any
provided prefix pattern, constructs ``W_pad`` shaped ``[..., new_obs_dim]`` with zeros in the last
``new_obs_dim - old_obs_dim`` columns, copies ``W``, and writes ``--output``.

Example::

    python3 expand_walk_checkpoint.py \\
      --input /path/to/walk/model_999.pt \\
      --output /path/to/kick/init_from_walk.pt \\
      --old_obs_dim 69 \\
      --new_obs_dim 86 \\
      --key_regex '^(module\\.)?(actor|critic)\\..*\\.weight$'

Dry-run::

    python3 expand_walk_checkpoint.py --input in.pt --output out.pt \\
      --old_obs_dim 54 --new_obs_dim 86 --dry_run

**Note.** Only **weight** tensors are padded. Biases, log-std vectors, recurrent states, privileged
critics with different widths, etc. must be checked manually after warm-start training.
Observation normalizers (if present under different keys / shapes) are **not** auto-patched.
"""

from __future__ import annotations

import argparse
import copy
import re

import torch


def pad_weights(
    state: dict[str, torch.Tensor],
    *,
    old_in: int,
    new_in: int,
    regex: re.Pattern[str],
    dry_run: bool,
) -> list[str]:
    if new_in <= old_in:
        raise ValueError("new_obs_dim must exceed old_obs_dim.")
    patched: list[str] = []

    for name, tensor in list(state.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim != 2 or tensor.shape[1] != old_in:
            continue
        if regex.search(name) is None:
            continue
        patched.append(f"{name} {tuple(tensor.shape)}")
        if dry_run:
            continue
        padded = tensor.new_zeros((tensor.shape[0], new_in))
        padded[:, :old_in] = tensor
        state[name] = padded

    return patched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--old_obs_dim", type=int, required=True)
    ap.add_argument("--new_obs_dim", type=int, required=True)
    ap.add_argument(
        "--state_key",
        default="model_state_dict",
        help="Field inside the `.pt` dict where weights reside.",
    )
    ap.add_argument(
        "--key_regex",
        default=r"^(module\.)?(actor|critic)\.0\.weight$",
        help="Regex applied against ``state_dict`` keys; must match weights only (omit bias/logstd). "
        r"Defaults to sequential MLP first layers for common RSL-RL layouts — adjust after inspection.",
    )
    ap.add_argument(
        "--regex_all_weights",
        action="store_true",
        help=(
            r"Overrides --key_regex to r'\\.weight\$' matched against tensors with "
            "`shape[1]==old_obs_dim` (pads every matching weight)."
        ),
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    regex = r".*\.weight$" if args.regex_all_weights else args.key_regex
    pattern = re.compile(regex)

    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    if args.state_key not in bundle:
        raise KeyError(f"Missing `{args.state_key}`; top-level keys: {list(bundle.keys())}")

    sd = bundle[args.state_key]
    if not isinstance(sd, dict):
        raise TypeError(f"`{args.state_key}` must be a dict[str, Tensor or Any], got {type(sd)}")

    out_bundle = copy.deepcopy(bundle)
    out_sd = {k: v for k, v in sd.items()}

    hit = pad_weights(out_sd, old_in=args.old_obs_dim, new_in=args.new_obs_dim, regex=pattern, dry_run=args.dry_run)
    print(f"[expand_walk_checkpoint] matched {len(hit)} tensors (old_in_dim={args.old_obs_dim});")
    for line in hit[:80]:
        print("  ", line)
    if len(hit) > 80:
        print(f"   ... (+{len(hit) - 80} more)")
    if not hit:
        print("  Showing first tensors for inspection:")
        for i, (k, v) in enumerate(out_sd.items()):
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {tuple(v.shape)}")
                if i > 49:
                    break

    out_bundle[args.state_key] = out_sd

    if args.dry_run:
        print("[expand_walk_checkpoint] dry_run: not writing.")
        return

    torch.save(out_bundle, args.output)
    print(f"[expand_walk_checkpoint] wrote {args.output}")


if __name__ == "__main__":
    main()
