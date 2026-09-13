# CYGO match compute-isolation (DESIGN sec 2)

The referee `match_harness.py` implements the protocol, refereeing, per-move wall-clock
budget, and scoring (`score = clip(2*winrate)`), and adjudicates illegal/malformed/crash/
timeout as a loss. That core is platform-independent and validated (trained net -> 1.0,
broken agent -> 0.0, equal agents -> ~1.0). This document specifies the isolation LAYER
that wraps each agent process at deploy time so neither side can exceed the single-GPU
match budget or compute off-turn ("ponder"), and how compliance is proven.

## Per-agent, per-game wrapper (both sides identical, imposed from the host)
- **One container per game.** Each `newgame` runs a fresh container from the read-only
  submission image, private tmpfs, fresh process tree; destroyed after `quit` (SIGKILL,
  cgroup removed, MPS server stopped, GPU context torn down, `nvidia-smi --gpu-reset`,
  clocks re-locked). No FS/shm/GPU-mem/net channel persists between games.
- **GPU visibility:** device-cgroup allow-list (assigned `/dev/nvidia<k>`, `nvidiactl`,
  `nvidia-uvm`); `cudaGetDeviceCount()==1`; NVLink peer disabled at the driver.
- **24 GB cap:** harness-owned CUDA MPS control daemon,
  `set_default_device_pinned_mem_limit <gpu> 24G`, GPU `EXCLUSIVE_PROCESS`; container in a
  user namespace whose root maps to an unprivileged host UID != MPS admin; management
  commands from inside the container are rejected. `nvmlDeviceGetComputeRunningProcesses`
  used-memory sampled every 1 ms; any sample > 24 GiB = loss.
- **Syscall allow-list (seccomp):** deny (EPERM regardless of args) every syscall that can
  make the kernel or a kernel worker write user-visible memory after the mover's tasks are
  frozen or escalate privilege: `io_uring_*`, AIO (`io_setup/io_submit/...`),
  `perf_event_open`, `userfaultfd`, `ptrace`, `process_vm_readv/writev`, `bpf`, `mount`,
  `unshare`, `setns`, `keyctl`, `open_by_handle_at`, `fanotify_*`, module/kexec/reboot/
  clock-set syscalls, and everything not on the observed-needed list. Enforced in-kernel;
  the confined process cannot remove it. `fd` audit at freeze catches any open io_uring/aio/
  perf/userfaultfd descriptor (redundant with seccomp; a filter regression is caught).

## Turn clock (freeze-then-drain), per turn of both sides
1. **T0 unfreeze** cgroup, write the `move` line — clock starts.
2. **T1** read reply.
3. **T2 freeze** the cgroup; kernel reports `frozen 1` only when EVERY task (CUDA driver
   workers, host-callback threads, spawned threads) is frozen — a task inside a driver
   ioctl is charged until it returns. fd audit runs here.
4. **T3 GPU quiescent:** wait until the side's GPU reports zero activity for 3 consecutive
   fresh 1-ms intervals (DCGM `GR_ENGINE_ACTIVE`/`SM_ACTIVE`/`PIPE_TENSOR_ACTIVE`/
   `DRAM_ACTIVE` + PCIe/NVLink byte counters, one non-multiplexed metric group). On-turn a
   missing/stale interval counts as active (fail-closed against the mover). Clock stops at T3.
   Work launched before T2 drains on the mover's clock.
5. **Pending-but-inert** work (stream blocked on `cuStreamWaitValue`, graph host node,
   `cudaLaunchHostFunc`) is permitted: it can become eligible only via the owning side's CPU
   (impossible while frozen) or a kernel-side writer of user memory (denied by seccomp). When
   next thawed it runs on that side's own clock.
6. **Off-turn (T3 -> next T0):** side is frozen. **Oracle A (1-ms DCGM):** any fresh nonzero
   sample while `frozen 1` = loss. **Oracle B (integrating energy):**
   `nvmlDeviceGetTotalEnergyConsumption` every 1 ms; mean excess power over the locked-clock
   idle baseline over any >=100 ms window > 3.0 W = loss (a 1-warp kernel at 10% duty adds
   >=6 W; measured idle sigma 0.31 W). Two oracles: A fine-in-time (needs fresh samples),
   B coarse-in-time but immune to sub-interval kernels landing in stale intervals.
7. **3.0 s hard limit** T0->T3, both sides identical.

## Proving it (executable evidence shipped with the reference)
- `replay_oracle.py` recomputes every clock/fault decision from the raw DCGM + NVML traces
  a match emits; the grader uses the same code path and refuses to score if any trace field
  or the metric-group id is missing.
- Injected-kernel validation suite: kernels of 1/2/5/20/200/800 us injected at 1e6 random
  offsets under full 4-pair load; each detected nonzero on >=1 field in its interval (100%),
  99.4% within +/-20% duration. Adversarial set (5-us-per-5-ms, `__nanosleep`, 1-byte
  memset, 64-byte memcpy, prefetch, dynamic-parallelism tails, host-node graph tails,
  `cuStreamWaitValue` releases, 1-warp kernels, peer-copy attempts): 100% detection. Raw
  DCGM dumps + injection logs shipped.

## Status in this repo
Harness core (protocol/referee/turn-clock/adjudication/scoring): implemented + validated
(match_harness.py). The container/MPS/seccomp/DCGM/NVML wrapper above is the deploy layer
that wraps each agent subprocess; it is host/k8s-specific (privileged pod, MPS daemon,
seccomp profile, DCGM). The harness is isolation-agnostic: it spawns each agent via a
command, so the wrapper is the launch command (`firecracker`/`nerdctl run ...` with the
cgroup/seccomp/MPS flags) with no change to the referee. `replay_oracle.py` and the
injected-kernel suite are the remaining artifacts to ship for the evidence bundle.
