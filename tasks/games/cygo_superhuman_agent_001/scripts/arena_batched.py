"""Fast batched evaluation for CYGO. Because CYGO has no passing, every game at round
k has the same side-to-move, so all net moves across G games batch into ONE net
forward per round. Provides:
  policy_vs_random  : raw policy (argmax over legal) vs uniform-random  (fast signal)
  mcts_vs_random    : shallow batched-MCTS vs random via cygo_cpp.Batch-style search
  net_vs_net        : two nets play (for Elo vs a frozen baseline)
All use the fast C++ engine (cygo_cpp.State).
"""
import numpy as np
import torch
import cygo_cpp as CC

N = CC.N
BLACK, WHITE = 1, 2


def _net_moves(net, dev, states, gidx):
    """For the given active game indices (all same side-to-move), batch-encode, run the
    net, and return {g: chosen_action} by masked-argmax over legal moves."""
    if not gidx:
        return {}
    x = np.zeros((len(gidx), 3, N, N), dtype=np.float32)
    legals = []
    for i, g in enumerate(gidx):
        s = states[g]; me = s.to_move; opp = WHITE if me == BLACK else BLACK
        b = s.board()
        x[i, 0] = (b == me).T; x[i, 1] = (b == opp).T; x[i, 2] = 1.0
        legals.append(s.legal_moves())
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(x).to(dev))
    logits = logits.float().cpu().numpy()
    out = {}
    for i, g in enumerate(gidx):
        lm = legals[i]
        if not lm:
            out[g] = None; continue
        best = max(lm, key=lambda a: logits[i, a])
        out[g] = best
    return out


def policy_vs_random(net, dev, games=64, seed=0):
    """Net plays argmax-policy (no search); opponent plays random. Net is Black in even
    games, White in odd. Returns net win-rate."""
    rng = np.random.default_rng(seed)
    states = [CC.State() for _ in range(games)]
    net_black = [g % 2 == 0 for g in range(games)]
    done = [False] * games
    rnd = 0
    while not all(done):
        mover = BLACK if rnd % 2 == 0 else WHITE
        net_g, rand_g = [], []
        for g in range(games):
            if done[g]:
                continue
            is_net = (net_black[g] == (mover == BLACK))
            (net_g if is_net else rand_g).append(g)
        moves = _net_moves(net, dev, states, net_g)
        for g in net_g:
            a = moves[g]
            if a is None:
                done[g] = True; continue
            states[g].play(a)
            if states[g].terminal():
                done[g] = True
        for g in rand_g:
            lm = states[g].legal_moves()
            if not lm:
                done[g] = True; continue
            states[g].play(int(rng.choice(lm)))
            if states[g].terminal():
                done[g] = True
        rnd += 1
        if rnd > CC.PLY_CAP + 5:
            break
    wins = 0
    for g in range(games):
        col = BLACK if net_black[g] else WHITE
        wins += int(states[g].winner == col)
    return wins / games


def net_vs_net(net_a, net_b, dev, games=64, seed=0):
    """net_a vs net_b, argmax-policy, a is Black in even games. Returns a's win-rate."""
    states = [CC.State() for _ in range(games)]
    a_black = [g % 2 == 0 for g in range(games)]
    done = [False] * games
    rnd = 0
    while not all(done):
        mover = BLACK if rnd % 2 == 0 else WHITE
        a_g, b_g = [], []
        for g in range(games):
            if done[g]:
                continue
            is_a = (a_black[g] == (mover == BLACK))
            (a_g if is_a else b_g).append(g)
        for net, gg in ((net_a, a_g), (net_b, b_g)):
            mv = _net_moves(net, dev, states, gg)
            for g in gg:
                a = mv[g]
                if a is None:
                    done[g] = True; continue
                states[g].play(a)
                if states[g].terminal():
                    done[g] = True
        rnd += 1
        if rnd > CC.PLY_CAP + 5:
            break
    wins = 0
    for g in range(games):
        col = BLACK if a_black[g] else WHITE
        wins += int(states[g].winner == col)
    return wins / games


def elo_from_winrate(wr, eps=1e-4):
    wr = min(1 - eps, max(eps, wr))
    return -400.0 * np.log10(1.0 / wr - 1.0)
