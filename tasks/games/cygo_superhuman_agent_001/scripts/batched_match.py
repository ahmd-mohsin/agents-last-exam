"""Fast batched CYGO match: G games in parallel between net_A and net_B, both playing
batched MCTS via cygo_cpp.Batch (one net forward per sim over all same-mover games).
Because CYGO has no passing, all active games share the side-to-move each round, so a
round splits cleanly into A-mover and B-mover groups. Distinct balanced openings + root
Dirichlet give game variety, so argmax-visit moves already yield a real win-rate.

Returns A's win-rate + Elo. --b-random makes B a uniform-random opponent (absolute anchor).
Usage: python3 batched_match.py --a <ckptA> [--b <ckptB> | --b-random] --games 60 --sims 128
"""
import argparse, math, numpy as np, torch
import cygo_cpp as CC
from azero import Net
from match_harness import make_openings
N = CC.N; BLACK, WHITE = 1, 2


def load(p, ch, blocks, dev):
    n = Net(N, ch=ch, blocks=blocks).to(dev).eval()
    s = torch.load(p, map_location=dev, weights_only=False)
    n.load_state_dict(s["net"] if "net" in s else s)
    return n


def run_sims(b, net, dev, sims):
    for _ in range(sims):
        x = b.collect()
        if x.shape[0] == 0:
            b.apply(np.zeros((0, N * N), np.float32), np.zeros((0,), np.float32)); continue
        with torch.no_grad():
            p, v = net(torch.from_numpy(x).to(dev))
        b.apply(p.float().cpu().numpy(), v.float().cpu().numpy())


def batched_match(netA, netB, dev, games, sims, noise, seed, b_random=False):
    rng = np.random.default_rng(seed)
    openings = make_openings(None, games, seed)
    states = [CC.State() for _ in range(games)]
    for g in range(games):
        for c, r in openings[g]:
            states[g].play(c * N + r)
    a_black = [g % 2 == 0 for g in range(games)]
    done = [s.terminal() for s in states]

    def move_group(net, sel, use_random):
        if not sel:
            return
        if use_random:
            for g in sel:
                lm = states[g].legal_moves()
                if not lm:
                    done[g] = True; continue
                states[g].play(int(rng.choice(lm))); done[g] = states[g].terminal()
            return
        b = CC.Batch(len(sel), sims, 1.5, 0.3, float(noise), 0)
        b.set_roots([states[g] for g in sel])
        run_sims(b, net, dev, sims)
        mv = b.best_moves()
        for i, g in enumerate(sel):
            a = mv[i]
            if a is None or a < 0:
                done[g] = True; continue
            states[g].play(int(a)); done[g] = states[g].terminal()

    guard = 0
    while not all(done) and guard < CC.PLY_CAP + 10:
        act = [g for g in range(games) if not done[g]]
        mover = states[act[0]].to_move
        a_sel = [g for g in act if a_black[g] == (mover == BLACK)]
        b_sel = [g for g in act if a_black[g] != (mover == BLACK)]
        move_group(netA, a_sel, False)
        move_group(netB, b_sel, b_random)
        guard += 1
    wins = sum(1 for g in range(games)
               if states[g].winner == (BLACK if a_black[g] else WHITE))
    return wins / games


def elo(wr):
    return -400 * math.log10(1 / min(0.999, max(0.001, wr)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", default=None)
    ap.add_argument("--b-random", action="store_true")
    ap.add_argument("--ch", type=int, default=256); ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--games", type=int, default=60); ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.25); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    netA = load(a.a, a.ch, a.blocks, dev)
    netB = None if a.b_random else load(a.b, a.ch, a.blocks, dev)
    import time; t0 = time.time()
    wr = batched_match(netA, netB, dev, a.games, a.sims, a.noise, a.seed, a.b_random)
    print(f"MATCH {a.label}: A_winrate={wr:.3f} ({a.games}g sims{a.sims}) "
          f"elo={elo(wr):+.0f} vs {'random' if a.b_random else a.b} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
