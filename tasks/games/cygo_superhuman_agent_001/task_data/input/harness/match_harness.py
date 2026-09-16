"""CYGO match referee (the gradeable core of the ALE harness).

Runs N games between a CANDIDATE agent program and the REFERENCE agent program, each
speaking the stdin/stdout protocol (see agent_proto.py). The referee holds the
authoritative engine, feeds the forced 6-ply opening, relays each mover the opponent's
last move, enforces the per-move wall-clock budget, and adjudicates: illegal move /
malformed reply / crash / clock violation = loss of that game. Score = clip(2*winrate).

This implements the protocol, refereeing, turn clock, and scoring. The DESIGN sec-2
COMPUTE ISOLATION layer (per-game container, MPS 24GB cap, seccomp allow-list,
freeze-then-drain, DCGM/NVML off-turn oracles) wraps EACH agent process at deploy time;
hooks are marked below. Here each agent is an isolated subprocess with a wall-clock
budget, which is the platform-independent core the isolation layer hardens.
"""
from __future__ import annotations
import argparse, json, random, select, subprocess, time
import cygo_cpp as CC

N = CC.N


class Agent:
    def __init__(self, cmd, color, seed):
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.first = True
        self._send(f"newgame {'black' if color==1 else 'white'} {seed}")
        if not self._read(20.0).startswith("= ok"):
            raise RuntimeError("agent did not ack newgame")

    def _send(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def _read(self, budget):
        """Read one reply line within `budget` seconds (wall clock). '' on timeout/EOF."""
        end = time.time() + budget
        while time.time() < end:
            if self.p.poll() is not None:
                return ""  # crashed/exited
            r, _, _ = select.select([self.p.stdout], [], [], max(0.0, end - time.time()))
            if r:
                return (self.p.stdout.readline() or "").strip()
        return ""  # timeout

    def move(self, opening, last_move, budget):
        pre = ("opening " + " ".join(f"{c} {r}" for c, r in opening) + " ") if self.first else ""
        self.first = False
        arg = "-" if last_move is None else f"{last_move[0]} {last_move[1]}"
        t0 = time.time()
        self._send(f"{pre}move {arg}")
        reply = self._read(budget)
        dt = time.time() - t0
        return reply, dt

    def close(self):
        try:
            self._send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(cand_cmd, ref_cmd, cand_color, opening, budget, seed):
    """Returns (winner_color, reason). Illegal/timeout/crash by a side => other wins."""
    st = CC.State()
    for c, r in opening:
        st.play(c * N + r)
    agents = {cand_color: Agent(cand_cmd, cand_color, seed),
              (3 - cand_color): Agent(ref_cmd, 3 - cand_color, seed)}
    last = None
    try:
        while not st.terminal():
            mover = st.to_move
            reply, dt = agents[mover].move(opening, last, budget)
            if dt > budget or not reply.startswith("="):
                return (3 - mover, f"{'timeout' if dt>budget else 'malformed'}({dt:.2f}s) by {mover}")
            toks = reply.split()
            if len(toks) != 3:
                return (3 - mover, f"badreply by {mover}: {reply!r}")
            c, r = int(toks[1]), int(toks[2])
            if not (0 <= c < N and 0 <= r < N) or not st.is_legal(c * N + r):
                return (3 - mover, f"illegal {c},{r} by {mover}")
            st.play(c * N + r)
            last = (c, r)
        return (st.winner, "terminal")
    finally:
        for ag in agents.values():
            ag.close()


def make_openings(pool, n, seed):
    """Deterministic legal 6-ply openings (random legal self-play prefixes)."""
    rng = random.Random(seed)
    outs = []
    for _ in range(n):
        s = CC.State(); mv = []
        for _ in range(6):
            lm = s.legal_moves()
            a = rng.choice(lm); s.play(a); mv.append((a // N, a % N))
        outs.append(mv)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="shell cmd launching the candidate agent")
    ap.add_argument("--reference", required=True, help="shell cmd launching the reference agent")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--budget", type=float, default=3.0, help="per-move wall-clock seconds")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    openings = make_openings(None, a.games, a.seed)
    wins = 0; recs = []
    t0 = time.time()
    for g in range(a.games):
        cand_color = 1 if g % 2 == 0 else 2   # candidate black in even games
        try:
            winner, reason = play_game(a.candidate, a.reference, cand_color, openings[g], a.budget, a.seed + g)
        except Exception as e:
            winner, reason = (3 - cand_color), f"candidate_error:{e}"
        win = int(winner == cand_color); wins += win
        recs.append({"game": g, "cand_color": cand_color, "winner": winner, "win": win, "reason": reason})
        if (g + 1) % 10 == 0:
            print(f"[match] {g+1}/{a.games} cand_wins={wins} ({wins/(g+1):.3f})", flush=True)
    wr = wins / a.games
    score = max(0.0, min(1.0, 2 * wr))
    out = {"games": a.games, "cand_winrate": wr, "score": score,
           "budget_s": a.budget, "wall_s": round(time.time() - t0, 1), "records": recs}
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
