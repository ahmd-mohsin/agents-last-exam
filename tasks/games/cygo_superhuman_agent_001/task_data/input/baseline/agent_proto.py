"""Reference CYGO agent speaking the match protocol on stdin/stdout (what a
submission's run.sh launches). One command per line:

  newgame <black|white> <seed>                         -> = ok
  [opening <c1> <r1> ... <c6> <r6>] move <oc> <or>|-   -> = <col> <row>
  quit

It maintains its own engine copy, applies the 6-ply opening and the opponent's moves,
and picks its move by MCTS with the loaded net (or policy/random for baselines). The
harness is the referee; this program only ever emits legal moves for its own turn.
"""
import sys, argparse, numpy as np, torch
import cygo_cpp as CC
from azero import Net
from arena_batched import _net_moves  # not used; kept for parity

N = CC.N


def load_net(path, ch, blocks, dev):
    net = Net(N, ch=ch, blocks=blocks).to(dev).eval()
    if path:
        st = torch.load(path, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"] if "net" in st else st)
    return net


def encode(state):
    b = state.board(); me = state.to_move; opp = 2 if me == 1 else 1
    x = np.zeros((1, 3, N, N), dtype=np.float32)
    x[0, 0] = (b == me).T; x[0, 1] = (b == opp).T; x[0, 2] = 1.0
    return x


def pick_move(net, dev, state, sims, mode, temp=0.0, noise=0.0, temp_plies=0):
    lm = state.legal_moves()
    if not lm:
        return None
    if mode == "random":
        return int(np.random.choice(lm))
    if mode == "mcts" and net is not None:
        # real single-position MCTS. For strength MEASUREMENT pass noise>0 (root
        # Dirichlet) + temp>0 (sample visits^(1/temp) for the first temp_plies) so games
        # are stochastic and win-rate reflects strength, not a deterministic mirror.
        b = CC.Batch(1, sims, 1.5, 0.3, float(noise), 0)
        b.set_root(state)
        for _ in range(sims):
            x = b.collect()
            if x.shape[0] == 0:
                b.apply(np.zeros((0, N * N), np.float32), np.zeros((0,), np.float32)); continue
            with torch.no_grad():
                p, v = net(torch.from_numpy(x).to(dev))
            b.apply(p.float().cpu().numpy(), v.float().cpu().numpy())
        if temp > 1e-3 and state.ply < temp_plies:
            vis = b.root_visits()
            w = np.zeros(N * N)
            for a in lm:
                w[a] = vis[a] ** (1.0 / temp)
            if w.sum() > 0:
                return int(np.random.choice(N * N, p=w / w.sum()))
        a = b.root_best_move()
        return a if a >= 0 else int(np.random.choice(lm))
    # policy argmax (net) — deterministic, always legal
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(encode(state)).to(dev))
    logits = logits[0].float().cpu().numpy()
    return max(lm, key=lambda a: logits[a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ch", type=int, default=256); ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--mode", choices=["policy", "mcts", "random"], default="policy")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--temp-plies", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = None if a.mode == "random" else load_net(a.weights, a.ch, a.blocks, dev)
    st = None; my_color = None
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newgame":
            my_color = 1 if parts[1] == "black" else 2
            if len(parts) > 2:
                np.random.seed(int(parts[2]) & 0x7fffffff)
            st = CC.State()
            print("= ok", flush=True)
        elif parts[0] == "quit":
            break
        else:
            # optional leading "opening c1 r1 ... c6 r6"
            i = 0
            if parts[0] == "opening":
                for k in range(6):
                    c, r = int(parts[1 + 2 * k]), int(parts[2 + 2 * k])
                    st.play(c * N + r)
                i = 13  # index of the 'move' token
            # now parts[i] == 'move'
            assert parts[i] == "move", f"bad cmd: {line!r}"
            opp = parts[i + 1]
            if opp != "-":
                oc, orr = int(opp), int(parts[i + 2])
                st.play(oc * N + orr)
            a_idx = pick_move(net, dev, st, a.sims, a.mode, a.temp, a.noise, a.temp_plies)
            if a_idx is None:
                print("= resign", flush=True); continue
            st.play(a_idx)
            print(f"= {a_idx // N} {a_idx % N}", flush=True)


if __name__ == "__main__":
    main()
