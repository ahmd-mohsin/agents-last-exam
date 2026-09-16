"""CYGO (Cylinder Gonnect) rules engine — reference implementation.

Board 13x13. Columns WRAP (cylinder); rows do not. Orthogonal adjacency. Go capture
(remove opponent groups with 0 liberties after placement), suicide illegal, positional
superko (no repeat of a full-board stone configuration). No passing. Win: after your
move, any of YOUR orthogonally-connected groups touches both row 0 and row 12 -> you win
immediately. A player with no legal move loses. Ply cap 600: if reached, more stones
wins, ties to White. Draws impossible.

Coordinates: point = (col, row), 0<=col<N (wraps), 0<=row<N (edges). Color: 1=Black
(moves first), 2=White. This is the correctness reference; a fast C++/bitboard engine
comes later. Deterministic; no policy here.
"""
from __future__ import annotations

import numpy as np

N = 13
PLY_CAP = 600
BLACK, WHITE, EMPTY = 1, 2, 0


def _zobrist(n, seed=20260911):
    rng = np.random.default_rng(seed)
    # [color-1, col, row] 64-bit keys; color in {black,white}
    return rng.integers(0, 2**63 - 1, size=(2, n, n), dtype=np.uint64)


class CygoState:
    def __init__(self, n=N):
        self.n = n
        self.board = np.zeros((n, n), dtype=np.int8)  # [col, row]
        self.to_move = BLACK
        self.ply = 0
        self.winner = 0  # 0 none, else BLACK/WHITE
        self._z = _zobrist(n)
        self._hash = np.uint64(0)
        self._history = {self._hash}  # positional superko set (board-only)

    # ---- geometry ----
    def neighbors(self, c, r):
        n = self.n
        yield (c - 1) % n, r
        yield (c + 1) % n, r
        if r > 0:
            yield c, r - 1
        if r < n - 1:
            yield c, r + 1

    def _group_and_libs(self, c, r, board=None):
        """Return (group set, liberty count) for the stone at (c,r)."""
        b = self.board if board is None else board
        color = b[c, r]
        stack = [(c, r)]
        group = {(c, r)}
        libs = set()
        while stack:
            cc, rr = stack.pop()
            for nc, nr in self.neighbors(cc, rr):
                v = b[nc, nr]
                if v == EMPTY:
                    libs.add((nc, nr))
                elif v == color and (nc, nr) not in group:
                    group.add((nc, nr))
                    stack.append((nc, nr))
        return group, len(libs)

    # ---- move legality + application ----
    def _would_capture(self, c, r, color, board):
        """Opponent groups adjacent to (c,r) that would have 0 libs after placement."""
        opp = WHITE if color == BLACK else BLACK
        captured = set()
        seen = set()
        for nc, nr in self.neighbors(c, r):
            if board[nc, nr] == opp and (nc, nr) not in seen:
                grp, libs = self._group_and_libs(nc, nr, board)
                seen |= grp
                if libs == 0:
                    captured |= grp
        return captured

    def legal(self, c, r):
        """Is placing to_move's stone at (c,r) legal? Returns (ok, resulting_hash,
        captured_set) — resulting_hash valid only if ok."""
        if self.board[c, r] != EMPTY or self.winner:
            return False, None, None
        color = self.to_move
        b = self.board.copy()
        b[c, r] = color
        captured = self._would_capture(c, r, color, b)
        for (cc, rr) in captured:
            b[cc, rr] = EMPTY
        # suicide check: own group must have >=1 liberty after captures
        _, libs = self._group_and_libs(c, r, b)
        if libs == 0:
            return False, None, None
        # positional superko: board-only hash must be new
        h = self._hash_of(b)
        if h in self._history:
            return False, None, None
        return True, h, captured

    def _hash_of(self, board):
        h = np.uint64(0)
        bs = np.argwhere(board == BLACK)
        ws = np.argwhere(board == WHITE)
        for c, r in bs:
            h ^= self._z[0, c, r]
        for c, r in ws:
            h ^= self._z[1, c, r]
        return h

    def legal_moves(self):
        if self.winner:
            return []
        out = []
        for c in range(self.n):
            for r in range(self.n):
                if self.board[c, r] == EMPTY:
                    ok, _, _ = self.legal(c, r)
                    if ok:
                        out.append((c, r))
        return out

    def _touches_both_edges(self, c, r, board):
        grp, _ = self._group_and_libs(c, r, board)
        rows = {rr for _, rr in grp}
        return (0 in rows) and (self.n - 1 in rows)

    def play(self, c, r):
        """Apply a legal move for to_move. Updates winner/terminal. Raises on illegal."""
        ok, h, captured = self.legal(c, r)
        if not ok:
            raise ValueError(f"illegal move {(c, r)} for {self.to_move}")
        color = self.to_move
        self.board[c, r] = color
        for (cc, rr) in captured:
            self.board[cc, rr] = EMPTY
        self._hash = h
        self._history.add(h)
        self.ply += 1
        opp = WHITE if color == BLACK else BLACK
        # win by connection (mover's group through (c,r)). Switch to_move anyway so
        # that at every terminal the side-to-move is NOT the connection winner.
        if self._touches_both_edges(c, r, self.board):
            self.winner = color
            self.to_move = opp
            return
        # switch side; opponent to move
        self.to_move = opp
        # opponent has no legal move -> opponent loses
        if not self.legal_moves():
            self.winner = color
            return
        # ply cap adjudication
        if self.ply >= PLY_CAP:
            nb = int((self.board == BLACK).sum())
            nw = int((self.board == WHITE).sum())
            self.winner = BLACK if nb > nw else WHITE  # ties -> White
            return

    def terminal(self):
        return self.winner != 0

    def clone(self):
        s = CygoState.__new__(CygoState)
        s.n = self.n
        s.board = self.board.copy()
        s.to_move = self.to_move
        s.ply = self.ply
        s.winner = self.winner
        s._z = self._z  # shared read-only tables
        s._hash = self._hash
        s._history = set(self._history)
        return s
