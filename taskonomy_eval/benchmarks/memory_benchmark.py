import argparse, json, time
from typing import Dict, List, Optional, Sequence, Tuple
import torch

from .repro.scheduler_repro import InterferenceAwareScheduler, SchedulerConfig

# --- Synthetic data (same shape/flow as wall-clock repro) ---
def generate_synthetic_data(num_steps:int, num_tasks:int, gradient_dim:int,
                            device:torch.device, seed:int)->Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    gradients = torch.randn(num_steps, num_tasks, gradient_dim, device=device)
    losses = torch.rand(num_steps, num_tasks, device=device) + 0.1
    return gradients, losses

# --- Very small aggregator shims (match the repro benchmark API/behavior) ---
def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.numel() == 1: return torch.ones_like(v)
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0) - 1
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = sorted_v - cssv / ind > 0
    nonzeros = torch.nonzero(cond, as_tuple=False)
    if nonzeros.numel() == 0: theta = cssv[-1] / v.numel()
    else:
        rho = nonzeros[-1, 0]
        theta = cssv[rho] / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    s = w.sum()
    return (torch.ones_like(v) / v.numel()) if s <= 0 else (w / s)

class BaseAgg:
    def __init__(self, K:int, device:torch.device): self.K, self.device = K, device
    def reset(self): pass
    def combine(self, grads:torch.Tensor, losses:Optional[torch.Tensor]=None,
                active_indices:Optional[torch.Tensor]=None)->torch.Tensor:
        raise NotImplementedError

class UniformAgg(BaseAgg):
    def combine(self, grads, losses=None, active_indices=None): return grads.mean(dim=0)

class PCGradAgg(BaseAgg):
    def combine(self, grads, losses=None, active_indices=None):
        g = grads.clone()
        n = g.shape[0]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                dot = torch.dot(g[i], grads[j])
                if dot < 0:
                    denom = grads[j].dot(grads[j]) + 1e-8
                    g[i] = g[i] - (dot/denom) * grads[j]
        return g.mean(dim=0)

class MGDAAgg(BaseAgg):
    def __init__(self, K, device, max_iter=50, lr=None):
        super().__init__(K, device); self.max_iter, self.lr = max_iter, lr
    def _solve_weights(self, grads:torch.Tensor)->torch.Tensor:
        m = grads.shape[0]
        if m==1: return torch.ones(1, device=self.device)
        gram = grads @ grads.t()
        mx = gram.abs().max().item()
        step = self.lr if self.lr is not None else 1.0/(mx+1e-6)
        w = torch.full((m,), 1.0/m, device=self.device)
        for _ in range(self.max_iter):
            gw = gram @ w
            w = project_to_simplex(w - step*gw)
        return w
    def combine(self, grads, losses=None, active_indices=None):
        w = self._solve_weights(grads.to(self.device))
        return torch.matmul(w, grads.to(self.device))

class GradNormAgg(BaseAgg):
    def __init__(self, K, device, alpha=0.5, lr=0.01):
        super().__init__(K, device); self.alpha, self.lr = alpha, lr; self.reset()
    def reset(self):
        self.weights = torch.ones(self.K, device=self.device)/self.K
        self.initial_losses = torch.zeros(self.K, device=self.device)
        self.initialized = torch.zeros(self.K, dtype=torch.bool, device=self.device)
    def combine(self, grads, losses=None, active_indices=None):
        if losses is None or active_indices is None:
            raise ValueError("GradNorm requires losses+active_indices")
        idx = active_indices.long()
        if (~self.initialized[idx]).any():
            m = ~self.initialized[idx]
            self.initial_losses[idx[m]] = losses[m].clamp_min(1e-6)
            self.initialized[idx[m]] = True
        ref = self.initial_losses[idx].clamp_min(1e-6)
        ratio = (losses.clamp_min(1e-6)/ref).pow(self.alpha)
        norms = grads.norm(dim=1).clamp_min(1e-8)
        tgt = norms.mean() * ratio
        upd = torch.exp(self.lr * (norms - tgt))
        self.weights[idx] = self.weights[idx] * upd
        self.weights = project_to_simplex(self.weights)
        w_sub = self.weights[idx]
        w_sub = w_sub / (w_sub.sum() + 1e-8)
        return torch.matmul(w_sub, grads)

class CAGradAgg(BaseAgg):
    def __init__(self, K, device, c=0.4):
        super().__init__(K, device); self.c = c; self.mgda = MGDAAgg(K, device)
    def reset(self): self.mgda = MGDAAgg(self.K, self.device)
    def combine(self, grads, losses=None, active_indices=None):
        grads = grads.to(self.device)
        w = self.mgda._solve_weights(grads)
        g_mgda = torch.matmul(w, grads)
        g_mean = grads.mean(dim=0)
        return (g_mgda + self.c*g_mean) / (1.0 + self.c)

class AdaTaskAgg(BaseAgg):
    def __init__(self, K, device, alpha=0.5):
        super().__init__(K, device); self.alpha = alpha; self.reset()
    def reset(self): self.eta = torch.ones(self.K, device=self.device)/self.K
    def combine(self, grads, losses=None, active_indices=None):
        if active_indices is None: raise ValueError("AdaTask needs active_indices")
        idx = active_indices.long()
        norms = grads.norm(dim=1)
        cap = torch.clamp(norms, max=10.0 / max(self.alpha, 1e-6))
        upd = torch.exp(self.alpha * cap)
        self.eta[idx] = self.eta[idx] * upd
        self.eta = torch.clamp(self.eta, min=1e-12); self.eta /= self.eta.sum()
        w_sub = self.eta[idx]; w_sub = w_sub / (w_sub.sum() + 1e-8)
        return torch.matmul(w_sub, grads)

def make_agg(name:str, K:int, device:torch.device)->BaseAgg:
    if name in {"uniform","scheduler"}: return UniformAgg(K, device)
    if name in {"gradnorm","scheduler_gradnorm"}: return GradNormAgg(K, device)
    if name in {"pcgrad","scheduler_pcgrad"}: return PCGradAgg(K, device)
    if name == "mgda": return MGDAAgg(K, device)
    if name == "cagrad": return CAGradAgg(K, device)
    if name in {"adatask","scheduler_adatask"}: return AdaTaskAgg(K, device)
    raise ValueError(f"unknown method {name}")

def mem_stats():
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }

def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize()

def run_once(gradients:torch.Tensor, losses:torch.Tensor, method:str,
             history_length:int, update_frequency:int, device:torch.device,
             sched_technique:str)->Dict[str, float]:
    K = gradients.shape[1]
    steps = gradients.shape[0]
    agg = make_agg(method, K, device)
    # Configure scheduler identical to wall-clock harness
    cfg = SchedulerConfig(history_length=history_length)
    cfg.update_frequency = update_frequency
    cfg.enable_gradient_accumulation = False
    cfg.enable_min_visit_constraint = False
    cfg.enable_warmup = False
    cfg.enable_loss_normalization = False

    # Technique toggles for SON-GOKU internals (use your repro scheduler flags)
    # exact: full-SVD path; jl: randomized low-rank; fd/incr: incremental PCA; edge: subsample edges (approx via pair mask).
    if sched_technique == "exact":
        cfg.enable_subspace_scheduling = False
        cfg.enable_incremental_pca = False
        cfg.enable_gram_trick = False
    elif sched_technique == "jl":
        cfg.enable_subspace_scheduling = True
        cfg.enable_incremental_pca = False
        cfg.enable_gram_trick = False
        cfg.svd_niter = 1
    elif sched_technique == "fd" or sched_technique == "incr":
        cfg.enable_subspace_scheduling = True
        cfg.enable_incremental_pca = True
        cfg.enable_gram_trick = True
    elif sched_technique == "gram":
        cfg.enable_subspace_scheduling = True
        cfg.enable_incremental_pca = False
        cfg.enable_gram_trick = True
    elif sched_technique == "edge":
        # We'll approximate by updating schedule less often to simulate sparse edge eval;
        # if you later add an explicit edge-sampling flag, wire it here.
        cfg.update_frequency = max(4, update_frequency)
    else:
        pass

    scheduler = InterferenceAwareScheduler([f"task_{i}" for i in range(K)], cfg)

    sink = 0.0
    reset_cuda_peak()
    synchronize(device)
    t0 = time.perf_counter()
    if method.startswith("scheduler") or method == "scheduler":
        # with scheduling: push grads each step, query active indices, then aggregate
        steps_per_epoch = max(1, steps // 5)
        for step in range(steps):
            g = gradients[step]; l = losses[step]
            grad_dict = {f"task_{i}": g[i] for i in range(K)}
            scheduler.update_gradients(grad_dict)
            epoch = step // steps_per_epoch
            active = scheduler.get_active_tasks(step, epoch)
            if not active: continue
            idx = torch.tensor([int(a.split("_")[1]) for a in active],
                               device=device, dtype=torch.long)
            combined = agg.combine(g[idx], l[idx], idx)
            sink += combined.norm().item()
    else:
        # no scheduling: aggregate all every step
        for step in range(steps):
            g = gradients[step]; l = losses[step]
            idx = torch.arange(K, device=device)
            combined = agg.combine(g, l, idx)
            sink += combined.norm().item()
    synchronize(device)
    t1 = time.perf_counter()

    stats = mem_stats()
    stats.update({"elapsed_sec": t1 - t0, "sink": float(sink)})
    return stats

def main():
    ap = argparse.ArgumentParser("Memory + wall-clock for synthetic MTL methods (repro setup).")
    ap.add_argument("--num-tasks","-k", type=int, default=6)
    ap.add_argument("--history-length","-r", type=int, default=10)
    ap.add_argument("--update-frequency","-u", type=int, default=100)
    ap.add_argument("--gradient-dim", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--methods", nargs="+",
                    default=["scheduler","uniform","gradnorm","mgda","pcgrad","cagrad","adatask",
                             "scheduler_adatask","scheduler_gradnorm","scheduler_pcgrad"])
    ap.add_argument("--techniques", nargs="+", default=["exact","jl","fd","incr","gram","edge"],
                    help="Scheduler-internal optimizations to test (ignored for non-scheduler methods)")
    ap.add_argument("--output-json", type=str, default="memory_benchmark_results.json")
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device=="cuda" or
              (args.device=="auto" and torch.cuda.is_available())) else torch.device("cpu"))

    gradients, losses = generate_synthetic_data(args.num_steps, args.num_tasks,
                                                args.gradient_dim, device, args.seed)

    results: Dict[str, Dict] = {}
    for tech in args.techniques:
        tech_block: Dict[str, Dict] = {}
        for method in args.methods:
            stats_runs: List[Dict] = []
            for _ in range(args.repeats):
                stats = run_once(gradients, losses, method,
                                 history_length=args.history_length,
                                 update_frequency=args.update_frequency,
                                 device=device, sched_technique=tech)
                stats_runs.append(stats)
            # aggregate
            import numpy as np
            elapseds = np.array([s["elapsed_sec"] for s in stats_runs], dtype=float)
            max_alloc = np.array([s["max_allocated"] for s in stats_runs], dtype=float)
            max_res = np.array([s["max_reserved"] for s in stats_runs], dtype=float)
            tech_block[method] = {
                "elapsed_mean_sec": float(elapseds.mean()),
                "elapsed_std_sec": float(elapseds.std(ddof=0)),
                "max_allocated_bytes_mean": float(max_alloc.mean()),
                "max_allocated_bytes_std": float(max_alloc.std(ddof=0)),
                "max_reserved_bytes_mean": float(max_res.mean()),
                "max_reserved_bytes_std": float(max_res.std(ddof=0)),
                "repeats": args.repeats,
            }
        results[tech] = tech_block

    with open(args.output_json, "w") as f:
        json.dump({
            "setup": {
                "num_tasks": args.num_tasks,
                "history_length": args.history_length,
                "update_frequency": args.update_frequency,
                "gradient_dim": args.gradient_dim,
                "num_steps": args.num_steps,
                "repeats": args.repeats,
                "device": str(device),
            },
            "results": results
        }, f, indent=2)

    # Pretty print
    print(f"Saved memory+time results to {args.output_json}")
    for tech in args.techniques:
        print(f"\n=== TECHNIQUE: {tech} ===")
        for m, st in results[tech].items():
            print(f"{m:20s}  time {st['elapsed_mean_sec']:.4f}s ± {st['elapsed_std_sec']:.4f}s"
                  f" | max_alloc {st['max_allocated_bytes_mean']/1e6:.1f}±{st['max_allocated_bytes_std']/1e6:.1f} MB"
                  f" | max_res {st['max_reserved_bytes_mean']/1e6:.1f}±{st['max_reserved_bytes_std']/1e6:.1f} MB")

if __name__ == "__main__":
    main()
