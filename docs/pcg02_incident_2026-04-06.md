# pcg02 Incident Notes (2026-04-06)

This note records the current RTS project state after the failed `Pixiu` 30k comparison run on `pcg02`.

## Current status

- No active RTS-related training processes remain on `pcg02`.
- `nvidia-smi` is responsive again.
- GPU `5` is still in an error state and should not be used.
- User plans to continue work on Windows first.

Checked on `2026-04-06 15:02 JST`.

## Relevant logs

- `2dgs` training log: `/home/wangyy/data_download/gsrelight_runs/logs/2dgs_pixiu_30k.log`
- `gs3` training log: `/home/wangyy/data_download/gsrelight_runs/logs/gs3_pixiu_30k.log`
- `2dgs` output dir: `/home/wangyy/data_download/gsrelight_runs/2dgs/NRHints/gogo/Pixiu`
- `gs3` output dir: `/home/wangyy/data_download/gsrelight_runs/gs3/NRHints/gogo/Pixiu`

## What was launched

- `gs3` was launched on physical GPU `3`.
- `2dgs` was launched on physical GPU `5`.
- Both runs targeted `NRHints/Pixiu/Pixiu`.
- Both runs were configured for `30000` iterations.

## 2dgs failure summary

The `2dgs` run failed at step `0`, before entering meaningful training progress.

Observed crash from `2dgs_pixiu_30k.log`:

```text
Training progress:   0%|          | 0/30000 [00:00<?, ?it/s]
...
File "/home/wangyy/RTS/2dgs/gaussian_renderer/__init__.py", line 287, in _compute_shadow_pass
    shadow_rendered, _, _, _ = view_rasterizer(
...
File "/home/wangyy/RTS/gs3/submodules/v_3dgs/v_3dgs/__init__.py", line 90, in forward
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, final_T = _C.rasterize_gaussians(*args)
RuntimeError: CUDA error: operation not supported on global/shared address space
```

The failing path is the `2dgs` shadow image pass that calls `gs3/submodules/v_3dgs`.

## Kernel / driver evidence

Relevant `dmesg` lines:

```text
[Mon Apr  6 03:21:37 2026] NVRM: Xid (PCI:0000:97:00): 13, pid=4067793, name=python, Graphics Exception: channel 0x0000000a, Class 0000cbc0, Offset 00000000, Data 00000000
[Mon Apr  6 03:22:07 2026] NVRM: Xid (PCI:0000:97:00): 31, pid=4067793, name=python, channel 0x0000000a, intr 00000000. MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_13 faulted @ 0x7f81_d0ab1000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
[Mon Apr  6 07:25:23 2026] NVRM: Xid (PCI:0000:97:00): 119, pid=5220, name=cache_mgr_main, Timeout after 45s of waiting for RPC response from GPU6 GSP! Expected function 76 (GSP_RM_CONTROL) sequence 3661611 (0x208f1105 0x8).
[Mon Apr  6 07:25:24 2026] NVRM: Xid (PCI:0000:97:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)
```

`pid=4067793` was the `2dgs` training process.

## Current GPU state

Latest `nvidia-smi` snapshot at `2026-04-06 15:02:41 JST`:

```text
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:97:00.0 N/A |                  N/A |
|ERR!   50C    P0            N/A  /  N/A  |       0MiB /  81559MiB |     N/A      Default |
|                                         |                        |                 ERR! |
```

Notes:

- `nvidia-smi` itself is now responsive.
- No running GPU processes are currently shown.
- GPU `5` remains unhealthy and should be avoided until reset or node recovery.

## Process status

Explicit check for RTS-related processes returned no active matches:

```text
ps -eo pid,ppid,stat,comm,args | grep -E '/home/wangyy/RTS/2dgs|/home/wangyy/RTS/gs3|python train.py -s /home/wangyy/data_download/gsrelight/NRHints/Pixiu/Pixiu|bash test.sh' | grep -v grep
```

The command returned exit code `1`, meaning no matching active process was found.

## Practical handoff

- Do not reuse GPU `5` on `pcg02`.
- If continuing on another machine, start from the existing logs and outputs listed above.
- For `2dgs`, the unstable path is the current shadow pass that calls `v_3dgs`.
- `gs3` itself did enter training setup successfully before the incident.
