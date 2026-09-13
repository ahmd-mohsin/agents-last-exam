"""Self-play using the C++ batched MCTS (cygo_cpp.Batch): tree+PUCT in C++, net eval
batched in torch. Drop-in replacement for azero_batched throughput. CLI = throughput
test / actor core."""
import argparse, time, numpy as np, torch
import cygo_cpp as CC
from azero import Net
N = CC.N


def play_batch(net, dev, n_games, sims, c_puct=1.5, temp_moves=30, max_moves=0):
    b = CC.Batch(n_games, sims, c_puct, 0.3, 0.25, temp_moves)
    while not b.all_done():
        if max_moves and b.move_no >= max_moves:
            break
        for _ in range(sims):
            x = b.collect()                      # (K,3,N,N) float32
            if x.shape[0] == 0:
                b.apply(np.zeros((0, N * N), np.float32), np.zeros((0,), np.float32)); continue
            xt = torch.from_numpy(x).to(dev)
            with torch.no_grad():
                p, v = net(xt)
            b.apply(p.float().cpu().numpy(), v.float().cpu().numpy())
        b.advance(1.0)
    return b.samples()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--max-moves", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(N, ch=a.ch, blocks=a.blocks).to(dev).eval()
    t0 = time.time()
    out = play_batch(net, dev, a.games, a.sims, max_moves=a.max_moves)
    dt = time.time() - t0
    pos = sum(len(g) for g in out)
    # sanity
    ok = all(all(abs(float(pi.sum()) - 1.0) < 1e-3 or pi.sum() == 0 for (_, pi, _) in g) for g in out)
    print(f"C++ MCTS self-play: {a.games} games sims={a.sims} -> {pos} positions in {dt:.1f}s "
          f"= {pos/dt:.0f} pos/s ; pi_normalized_ok={ok}")
