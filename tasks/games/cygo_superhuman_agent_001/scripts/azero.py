"""Minimal but real AlphaZero pipeline for CYGO. Proof-of-pipeline: validates that
self-play + a column-circular ResNet + PUCT MCTS learns to play (arena win-rate vs
random rises), with checkpoint/resume so it survives ephemeral-box restarts.

Scales by BOARD_N (use 7 for a fast proof, 13 for the real task). This is the pipeline
skeleton; the full 24-day run needs the C++ engine + batched GPU MCTS + aux heads.
"""
from __future__ import annotations

import argparse, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cygo_engine as E


# ---------------- network (column-circular = cylinder-equivariant) ----------------
class CircConv(nn.Module):
    """Conv that wraps columns (width) and zero-pads rows (height)."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, cout, k, padding=0)

    def forward(self, x):  # x: (B,C,H=row,W=col)
        p = self.k // 2
        x = F.pad(x, (p, p, 0, 0), mode="circular")   # wrap columns
        x = F.pad(x, (0, 0, p, p), mode="constant")   # zero-pad rows
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = CircConv(c, c); self.n1 = nn.BatchNorm2d(c)
        self.c2 = CircConv(c, c); self.n2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, n, ch=64, blocks=6, cin=3):
        super().__init__()
        self.n = n
        self.stem = nn.Sequential(CircConv(cin, ch), nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.ph = CircConv(ch, 2, 1); self.pn = nn.BatchNorm2d(2)
        self.pf = nn.Linear(2 * n * n, n * n)
        self.vh = CircConv(ch, 1, 1); self.vn = nn.BatchNorm2d(1)
        self.vf1 = nn.Linear(n * n, 64); self.vf2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.res(self.stem(x))
        p = F.relu(self.pn(self.ph(h))).flatten(1)
        p = self.pf(p)
        v = F.relu(self.vn(self.vh(h))).flatten(1)
        v = torch.tanh(self.vf2(F.relu(self.vf1(v)))).squeeze(-1)
        return p, v


def encode(state):
    """Side-relative planes: [my stones, opp stones, ones]. Layout (C, row, col)."""
    n = state.n
    me, opp = state.to_move, (E.WHITE if state.to_move == E.BLACK else E.BLACK)
    b = state.board
    x = np.zeros((3, n, n), dtype=np.float32)
    x[0] = (b == me).T.astype(np.float32)   # transpose: board is [col,row] -> [row,col]
    x[1] = (b == opp).T.astype(np.float32)
    x[2] = 1.0
    return x


def mv_idx(c, r, n):
    return c * n + r


# ---------------- MCTS (negamax; value from side-to-move perspective) -------------
class Node:
    __slots__ = ("state", "P", "N", "W", "children", "legal", "is_term")
    def __init__(self, state):
        self.state = state
        self.P = None; self.N = None; self.W = None
        self.children = {}
        self.legal = None
        self.is_term = state.terminal()


def _eval(net, state, dev):
    x = torch.from_numpy(encode(state)[None]).to(dev)
    with torch.no_grad():
        p, v = net(x)
    return p[0].float().cpu().numpy(), float(v[0])


def mcts(net, root_state, sims, dev, c_puct=1.5, dir_alpha=0.3, dir_eps=0.25, temp=1.0):
    n = root_state.n
    root = Node(root_state.clone())
    _expand(root, net, dev)
    if root.legal:  # dirichlet noise at root
        noise = np.random.dirichlet([dir_alpha] * len(root.legal))
        for i, a in enumerate(root.legal):
            root.P[a] = (1 - dir_eps) * root.P[a] + dir_eps * noise[i]
    for _ in range(sims):
        _simulate(root, net, dev, c_puct)
    visits = np.array([root.N[a] if root.N is not None else 0 for a in range(n * n)], dtype=np.float64)
    if visits.sum() == 0:
        # no sims expanded (shouldn't happen); fall back uniform over legal
        pi = np.zeros(n * n);
        for a in root.legal: pi[a] = 1.0 / len(root.legal)
        return pi, root
    if temp <= 1e-3:
        pi = np.zeros(n * n); pi[int(visits.argmax())] = 1.0
    else:
        v = visits ** (1.0 / temp)
        pi = v / v.sum()
    return pi, root


def _expand(node, net, dev):
    s = node.state
    n = s.n
    if node.is_term:
        return
    legal = s.legal_moves()
    node.legal = [mv_idx(c, r, n) for (c, r) in legal]
    p_logits, _ = _eval(net, s, dev)
    mask = np.full(n * n, -1e9, dtype=np.float32)
    for a in node.legal:
        mask[a] = p_logits[a]
    e = np.exp(mask - mask.max()); P = e / e.sum()
    node.P = P
    node.N = np.zeros(n * n, dtype=np.float64)
    node.W = np.zeros(n * n, dtype=np.float64)


def _simulate(node, net, dev, c_puct):
    s = node.state
    if node.is_term:
        # side to move at a terminal state has lost (or ply-cap: use winner)
        return 1.0 if s.winner == s.to_move else -1.0
    if node.legal is None:
        _expand(node, net, dev)
        _, v = _eval(net, s, dev)
        return -v  # value flips going up one ply
    # PUCT select
    sqrtN = math.sqrt(node.N.sum() + 1)
    best_a, best_u = None, -1e18
    for a in node.legal:
        q = (node.W[a] / node.N[a]) if node.N[a] > 0 else 0.0
        u = q + c_puct * node.P[a] * sqrtN / (1 + node.N[a])
        if u > best_u:
            best_u, best_a = u, a
    if best_a not in node.children:
        c, r = divmod(best_a, s.n)
        ns = s.clone(); ns.play(c, r)
        node.children[best_a] = Node(ns)
    v_child = _simulate(node.children[best_a], net, dev, c_puct)
    node.N[best_a] += 1
    node.W[best_a] += -v_child   # value of action a from THIS node's perspective
    return -v_child


# ---------------- self-play + training ----------------
def self_play_game(net, n, sims, dev, temp_moves=None):
    temp_moves = temp_moves if temp_moves is not None else n * 2
    s = E.CygoState(n)
    hist = []  # (encoded, pi, player)
    while not s.terminal():
        temp = 1.0 if s.ply < temp_moves else 1e-3
        pi, _ = mcts(net, s, sims, dev, temp=temp)
        hist.append((encode(s), pi.astype(np.float32), s.to_move))
        a = int(np.random.choice(len(pi), p=pi))
        c, r = divmod(a, n)
        s.play(c, r)
    data = []
    for enc, pi, player in hist:
        z = 1.0 if s.winner == player else -1.0
        data.append((enc, pi, np.float32(z)))
    return data, s


def arena_vs_random(net, n, sims, dev, games=20):
    """Net (MCTS) vs uniform-random legal mover. Returns net win-rate."""
    wins = 0
    for g in range(games):
        s = E.CygoState(n)
        net_is_black = (g % 2 == 0)
        while not s.terminal():
            net_turn = (s.to_move == E.BLACK) == net_is_black
            if net_turn:
                pi, _ = mcts(net, s, sims, dev, temp=1e-3, dir_eps=0.0)
                a = int(pi.argmax())
                c, r = divmod(a, n)
            else:
                c, r = random.choice(s.legal_moves())
            s.play(c, r)
        net_color = E.BLACK if net_is_black else E.WHITE
        wins += int(s.winner == net_color)
    return wins / games


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.board
    net = Net(n, ch=args.ch, blocks=args.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buf = []
    start_iter = 0
    ckpt = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        start_iter = st["iter"]; buf = st.get("buf", [])
        print(f"[resume] from iter {start_iter}, buf={len(buf)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a")

    for it in range(start_iter, args.iters):
        t0 = time.time()
        net.eval()
        new = 0
        for _ in range(args.games_per_iter):
            data, _ = self_play_game(net, n, args.sims, dev)
            buf.extend(data); new += len(data)
        buf = buf[-args.buf_cap:]
        # train
        net.train()
        for _ in range(args.steps_per_iter):
            idx = np.random.randint(0, len(buf), size=min(args.batch, len(buf)))
            xb = torch.from_numpy(np.stack([buf[i][0] for i in idx])).to(dev)
            pib = torch.from_numpy(np.stack([buf[i][1] for i in idx])).to(dev)
            zb = torch.from_numpy(np.array([buf[i][2] for i in idx], dtype=np.float32)).to(dev)
            p, v = net(xb)
            lp = -(pib * F.log_softmax(p, dim=1)).sum(1).mean()
            lv = F.mse_loss(v, zb)
            loss = lp + lv
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        wr = arena_vs_random(net, n, args.sims, dev, games=args.arena_games)
        dt = time.time() - t0
        msg = (f"iter {it+1}/{args.iters} newsamples={new} buf={len(buf)} "
               f"loss={loss.item():.3f} (p={lp.item():.3f} v={lv.item():.3f}) "
               f"arena_vs_random={wr:.2f} {dt:.0f}s")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "iter": it + 1, "buf": buf[-args.buf_cap:],
                    "cfg": vars(args)}, ckpt)
    print("done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=7)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--steps-per-iter", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buf-cap", type=int, default=100000)
    ap.add_argument("--arena-games", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="run_proof")
    ap.add_argument("--resume", action="store_true")
    train(ap.parse_args())
