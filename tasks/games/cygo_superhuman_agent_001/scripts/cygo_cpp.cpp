// CYGO fast engine (C++/pybind11). Correctness parity with cygo_engine.py; built for
// throughput (byte board + precomputed neighbors + flood fill). idx = c*N + r.
// Columns wrap (cylinder); rows don't. Go capture, suicide illegal, positional superko
// (Zobrist board hash), win by top<->bottom connection, no-move loss, ply-cap-600
// adjudication (more stones wins; ties -> White).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>

namespace py = pybind11;

static constexpr int N = 13;
static constexpr int NN = N * N;
static constexpr int PLY_CAP = 600;
static constexpr int8_t EMPTY = 0, BLACK = 1, WHITE = 2;

struct Tables {
    std::array<std::array<int, 4>, NN> nbr;   // up to 4 neighbors
    std::array<int, NN> nnbr;
    std::array<std::array<uint64_t, NN>, 2> z; // zobrist [color-1][idx]
    Tables() {
        for (int c = 0; c < N; ++c) for (int r = 0; r < N; ++r) {
            int i = c * N + r, k = 0;
            int cl = (c - 1 + N) % N, cr = (c + 1) % N;
            nbr[i][k++] = cl * N + r;
            nbr[i][k++] = cr * N + r;
            if (r > 0) nbr[i][k++] = c * N + (r - 1);
            if (r < N - 1) nbr[i][k++] = c * N + (r + 1);
            nnbr[i] = k;
        }
        std::mt19937_64 rng(20260911ULL);
        for (int col = 0; col < 2; ++col) for (int i = 0; i < NN; ++i) z[col][i] = rng();
    }
};
static const Tables T;

struct State {
    std::array<int8_t, NN> board;
    int8_t to_move;
    int ply;
    int8_t winner;
    uint64_t hash;
    std::unordered_set<uint64_t> history;

    State() { reset(); }
    void reset() {
        board.fill(EMPTY); to_move = BLACK; ply = 0; winner = 0; hash = 0;
        history.clear(); history.insert(0);
    }

    // early-exit: does the group containing idx have >=1 liberty?
    bool has_liberty(int idx, const std::array<int8_t, NN>& b) const {
        int8_t color = b[idx];
        std::array<bool, NN> seen{};
        int stack[NN]; int sp = 0; stack[sp++] = idx; seen[idx] = true;
        while (sp) {
            int cur = stack[--sp];
            for (int k = 0; k < T.nnbr[cur]; ++k) {
                int nb = T.nbr[cur][k]; int8_t v = b[nb];
                if (v == EMPTY) return true;
                if (v == color && !seen[nb]) { seen[nb] = true; stack[sp++] = nb; }
            }
        }
        return false;
    }

    // flood fill group of idx in `b`; fill grp; return liberty count
    int group_libs(int idx, const std::array<int8_t, NN>& b, std::vector<int>& grp) const {
        int8_t color = b[idx];
        grp.clear();
        std::array<bool, NN> seen{}; // false
        std::vector<int> stack; stack.push_back(idx); seen[idx] = true;
        int libs = 0; std::array<bool, NN> libseen{};
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back(); grp.push_back(cur);
            for (int k = 0; k < T.nnbr[cur]; ++k) {
                int nb = T.nbr[cur][k]; int8_t v = b[nb];
                if (v == EMPTY) { if (!libseen[nb]) { libseen[nb] = true; ++libs; } }
                else if (v == color && !seen[nb]) { seen[nb] = true; stack.push_back(nb); }
            }
        }
        return libs;
    }

    // try move at idx for to_move on a scratch board; return (ok, resulting_hash).
    // captured filled with removed points.
    bool try_move(int idx, std::array<int8_t, NN>& b, uint64_t& out_hash,
                  std::vector<int>& captured) const {
        if (board[idx] != EMPTY || winner) return false;
        int8_t color = to_move, opp = (color == BLACK ? WHITE : BLACK);
        b = board;
        b[idx] = color;
        captured.clear();
        std::array<bool, NN> done{};
        std::vector<int> grp;
        for (int k = 0; k < T.nnbr[idx]; ++k) {
            int nb = T.nbr[idx][k];
            if (b[nb] == opp && !done[nb]) {
                if (has_liberty(nb, b)) continue;   // alive: cheap early-exit, common case
                group_libs(nb, b, grp);              // dead: collect cells to remove
                for (int g : grp) { done[g] = true; captured.push_back(g); }
            }
        }
        for (int g : captured) b[g] = EMPTY;
        if (!has_liberty(idx, b)) return false;      // suicide
        // incremental board-only hash: add placed stone, remove captured opp stones
        int ci = (color == BLACK) ? 0 : 1, oi = (opp == BLACK) ? 0 : 1;
        uint64_t h = hash ^ T.z[ci][idx];
        for (int g : captured) h ^= T.z[oi][g];
        if (history.count(h)) return false;         // positional superko
        out_hash = h;
        return true;
    }

    bool is_legal(int idx) const {
        std::array<int8_t, NN> b; uint64_t h; std::vector<int> cap;
        return try_move(idx, b, h, cap);
    }

    std::vector<int> legal_moves() const {
        std::vector<int> out;
        if (winner) return out;
        std::array<int8_t, NN> b; uint64_t h; std::vector<int> cap;
        for (int i = 0; i < NN; ++i)
            if (board[i] == EMPTY && try_move(i, b, h, cap)) out.push_back(i);
        return out;
    }

    bool touches_both_edges(int idx, const std::array<int8_t, NN>& b) const {
        std::vector<int> grp; group_libs(idx, b, grp);
        bool top = false, bot = false;
        for (int g : grp) { int r = g % N; if (r == 0) top = true; if (r == N - 1) bot = true; }
        return top && bot;
    }

    void play(int idx) {
        std::array<int8_t, NN> b; uint64_t h; std::vector<int> cap;
        if (!try_move(idx, b, h, cap)) throw std::runtime_error("illegal move");
        board = b; hash = h; history.insert(h); ++ply;
        int8_t color = to_move, opp = (color == BLACK ? WHITE : BLACK);
        if (touches_both_edges(idx, board)) { winner = color; to_move = opp; return; }
        to_move = opp;
        if (legal_moves().empty()) { winner = color; return; }
        if (ply >= PLY_CAP) {
            int nb = 0, nw = 0;
            for (int i = 0; i < NN; ++i) { if (board[i] == BLACK) ++nb; else if (board[i] == WHITE) ++nw; }
            winner = (nb > nw) ? BLACK : WHITE;
        }
    }

    bool terminal() const { return winner != 0; }

    py::array_t<int8_t> board_np() const {
        auto a = py::array_t<int8_t>({N, N});   // [col, row]
        auto v = a.mutable_unchecked<2>();
        for (int c = 0; c < N; ++c) for (int r = 0; r < N; ++r) v(c, r) = board[c * N + r];
        return a;
    }

    State clone() const { return *this; }
};

// ------------------------- batched MCTS (tree + PUCT in C++) -------------------------
// G games run in lockstep; Python drives net eval. Usage per move:
//   x = b.collect()                 # (K,3,N,N) float32 of leaves needing eval (K<=active)
//   pri, val = net(x)               # Python/torch
//   b.apply(pri, val)               # expand+backup; repeat `sims` times
//   b.advance(temp)                 # pick moves, record samples, re-root; increments move_no
//   ... until b.all_done(); b.samples() -> per-game [(enc(3,N,N), pi(NN), player)]
struct MNode {
    State st;
    bool expanded = false, term = false, noised = false;
    std::vector<int> legal;
    std::array<float, NN> P;
    std::array<double, NN> W;
    std::array<int, NN> Nv;
    std::unordered_map<int, int> ch;   // action -> pool index
};

struct Sample { std::array<float, 3 * NN> enc; std::array<float, NN> pi; int player; };

static std::mt19937_64 g_rng(12345ULL);

static void encode_into(const State& s, float* out) {  // (3,N,N) [plane,row,col]
    int me = s.to_move, opp = (me == BLACK ? WHITE : BLACK);
    for (int i = 0; i < 3 * NN; ++i) out[i] = 0.f;
    for (int c = 0; c < N; ++c) for (int r = 0; r < N; ++r) {
        int8_t v = s.board[c * N + r];
        int rc = r * N + c;
        if (v == me) out[rc] = 1.f;
        else if (v == opp) out[NN + rc] = 1.f;
        out[2 * NN + rc] = 1.f;
    }
}

class Batch {
public:
    int G, sims, cur_sim = 0, move_no = 0;
    float c_puct, dir_alpha, dir_eps; int temp_moves;
    std::vector<std::vector<MNode>> pool;
    std::vector<int> rootidx;
    std::vector<bool> done;
    std::vector<std::vector<std::pair<int,int>>> path;  // per game edges (nodeidx, action)
    std::vector<int> leaf;         // per game leaf node idx this sim (-1 if none)
    std::vector<double> leafval;   // terminal value if leaf terminal
    std::vector<char> leaf_eval;   // 1 if leaf needs net eval this sim
    std::vector<std::vector<Sample>> hist;
    std::vector<int8_t> winners;

    Batch(int g, int s, float cp, float da, float de, int tm)
        : G(g), sims(s), c_puct(cp), dir_alpha(da), dir_eps(de), temp_moves(tm) {
        pool.resize(G); rootidx.assign(G, 0); done.assign(G, false);
        path.resize(G); leaf.assign(G, -1); leafval.assign(G, 0);
        leaf_eval.assign(G, 0); hist.resize(G); winners.assign(G, 0);
        for (int g2 = 0; g2 < G; ++g2) { pool[g2].emplace_back(); pool[g2][0].st = State(); }
    }

    bool all_done() { for (int g = 0; g < G; ++g) if (!done[g]) return false; return true; }

    // Seed game 0's tree to an arbitrary position (for single-position agent MCTS).
    // Clones the given State (preserving superko history). Use with n_games==1.
    void set_root(const State& s) {
        pool[0].clear(); pool[0].emplace_back(); pool[0][0].st = s;
        rootidx[0] = 0; done[0] = false; path[0].clear(); leaf[0] = -1; leaf_eval[0] = 0;
    }

    // Best root action of game 0 by visit count (after running sims). -1 if none.
    int root_best_move() {
        MNode& r = pool[0][rootidx[0]];
        if (!r.expanded || r.legal.empty()) return -1;
        int best = r.legal[0]; double bv = -1;
        for (int a : r.legal) if (r.Nv[a] > bv) { bv = r.Nv[a]; best = a; }
        return best;
    }

    // Seed ALL games from a list of States (batched agent MCTS for a match). len==G.
    void set_roots(py::list states) {
        for (int g = 0; g < G && g < (int)states.size(); ++g) {
            pool[g].clear(); pool[g].emplace_back();
            pool[g][0].st = states[g].cast<State>();
            rootidx[g] = 0; done[g] = false; path[g].clear();
            leaf[g] = -1; leaf_eval[g] = 0;
        }
    }

    // Argmax-visit move per game (after sims). -1 where no legal/expanded.
    py::list best_moves() {
        py::list out;
        for (int g = 0; g < G; ++g) {
            MNode& r = pool[g][rootidx[g]];
            int best = -1; double bv = -1;
            if (r.expanded) for (int a : r.legal) if (r.Nv[a] > bv) { bv = r.Nv[a]; best = a; }
            out.append(best);
        }
        return out;
    }

    // Root visit counts of game 0 (size NN), for temperature sampling in Python.
    py::array_t<double> root_visits() {
        auto a = py::array_t<double>(NN); auto v = a.mutable_unchecked<1>();
        MNode& r = pool[0][rootidx[0]];
        for (int i = 0; i < NN; ++i) v(i) = (r.expanded ? r.Nv[i] : 0.0);
        return a;
    }

    // descend from root to a leaf for one game; fill path[g], leaf[g], leaf_eval[g], leafval[g]
    void descend(int g) {
        path[g].clear();
        int idx = rootidx[g];
        while (true) {
            MNode& nd = pool[g][idx];
            if (nd.term) { leaf[g] = idx; leaf_eval[g] = 0;
                leafval[g] = (nd.st.winner == nd.st.to_move) ? 1.0 : -1.0; return; }
            if (!nd.expanded) { leaf[g] = idx; leaf_eval[g] = 1; return; }
            double sqrtN = 1.0; { double tot = 0; for (int a : nd.legal) tot += nd.Nv[a]; sqrtN = std::sqrt(tot + 1); }
            int best_a = nd.legal.empty() ? -1 : nd.legal[0]; double best_u = -1e18;
            for (int a : nd.legal) {
                double q = nd.Nv[a] > 0 ? nd.W[a] / nd.Nv[a] : 0.0;
                double u = q + c_puct * nd.P[a] * sqrtN / (1 + nd.Nv[a]);
                if (u > best_u) { best_u = u; best_a = a; }
            }
            path[g].push_back({idx, best_a});
            auto it = nd.ch.find(best_a);
            if (it == nd.ch.end()) {
                State ns = nd.st; ns.play(best_a);
                int ni = pool[g].size(); pool[g].emplace_back(); pool[g][ni].st = ns;
                // note: pool may reallocate; re-fetch nd not needed (we index by idx)
                pool[g][idx].ch[best_a] = ni;
                leaf[g] = ni; MNode& lf = pool[g][ni];
                if (lf.st.terminal()) { lf.term = true; leaf_eval[g] = 0;
                    leafval[g] = (lf.st.winner == lf.st.to_move) ? 1.0 : -1.0; }
                else leaf_eval[g] = 1;
                return;
            }
            idx = it->second;
        }
    }

    // returns numpy (K,3,N,N) of leaves needing eval; records order in eval_order
    std::vector<int> eval_order;
    py::array_t<float> collect() {
        eval_order.clear();
        for (int g = 0; g < G; ++g) {
            if (done[g]) { leaf[g] = -1; continue; }
            descend(g);
            if (leaf_eval[g]) eval_order.push_back(g);
        }
        int K = eval_order.size();
        auto arr = py::array_t<float>({K, 3, N, N});
        float* p = arr.mutable_data();
        for (int k = 0; k < K; ++k) {
            int g = eval_order[k];
            encode_into(pool[g][leaf[g]].st, p + (size_t)k * 3 * NN);
        }
        return arr;
    }

    void backup(int g, double v) {
        for (int e = (int)path[g].size() - 1; e >= 0; --e) {
            int nidx = path[g][e].first, a = path[g][e].second;
            MNode& nd = pool[g][nidx];
            nd.W[a] += -v; nd.Nv[a] += 1; v = -v;
        }
    }

    void apply(py::array_t<float, py::array::c_style | py::array::forcecast> priors,
               py::array_t<float, py::array::c_style | py::array::forcecast> values) {
        auto pr = priors.unchecked<2>(); auto vv = values.unchecked<1>();
        for (int k = 0; k < (int)eval_order.size(); ++k) {
            int g = eval_order[k]; MNode& lf = pool[g][leaf[g]];
            lf.legal = lf.st.legal_moves();
            double mx = -1e30; for (int a : lf.legal) mx = std::max(mx, (double)pr(k, a));
            double sum = 0; for (int a : lf.legal) { double e = std::exp(pr(k, a) - mx); lf.P[a] = (float)e; sum += e; }
            for (int a : lf.legal) lf.P[a] /= (float)sum;
            for (int a = 0; a < NN; ++a) { lf.W[a] = 0; lf.Nv[a] = 0; }
            // root Dirichlet noise (once)
            if (leaf[g] == rootidx[g] && !lf.noised && !lf.legal.empty()) {
                std::gamma_distribution<double> gd(dir_alpha, 1.0);
                std::vector<double> nz(lf.legal.size()); double ns = 0;
                for (size_t i = 0; i < lf.legal.size(); ++i) { nz[i] = gd(g_rng); ns += nz[i]; }
                for (size_t i = 0; i < lf.legal.size(); ++i)
                    lf.P[lf.legal[i]] = (1 - dir_eps) * lf.P[lf.legal[i]] + dir_eps * (float)(nz[i] / ns);
                lf.noised = true;
            }
            lf.expanded = true;
            backup(g, (double)vv(k));
        }
        // terminal leaves (no eval): backup their value
        for (int g = 0; g < G; ++g) {
            if (done[g] || leaf_eval[g]) continue;
            if (leaf[g] < 0) continue;
            backup(g, leafval[g]);
        }
        cur_sim++;
    }

    void advance(double temp) {
        for (int g = 0; g < G; ++g) {
            if (done[g]) continue;
            MNode& root = pool[g][rootidx[g]];
            if (root.legal.empty()) { done[g] = true; winners[g] = root.st.winner; continue; }
            // visit counts
            std::array<double, NN> vis{}; double tot = 0;
            for (int a : root.legal) { vis[a] = root.Nv[a]; tot += vis[a]; }
            // record sample (pi = normalized visits)
            Sample sp; encode_into(root.st, sp.enc.data());
            for (int a = 0; a < NN; ++a) sp.pi[a] = tot > 0 ? (float)(vis[a] / tot) : 0.f;
            sp.player = root.st.to_move; hist[g].push_back(sp);
            // choose action
            int a_sel;
            if (tot == 0) { a_sel = root.legal[g_rng() % root.legal.size()]; }
            else if (move_no < temp_moves && temp > 1e-3) {
                std::vector<double> w(root.legal.size()); double s = 0;
                for (size_t i = 0; i < root.legal.size(); ++i) { w[i] = std::pow(vis[root.legal[i]], 1.0 / temp); s += w[i]; }
                double rdraw = std::uniform_real_distribution<double>(0, s)(g_rng), acc = 0; a_sel = root.legal.back();
                for (size_t i = 0; i < root.legal.size(); ++i) { acc += w[i]; if (rdraw <= acc) { a_sel = root.legal[i]; break; } }
            } else {
                a_sel = root.legal[0]; double bv = -1;
                for (int a : root.legal) if (vis[a] > bv) { bv = vis[a]; a_sel = a; }
            }
            // advance state, rebuild tree with fresh root
            State ns = root.st; ns.play(a_sel);
            pool[g].clear(); pool[g].emplace_back(); pool[g][0].st = ns; rootidx[g] = 0;
            if (ns.terminal()) { done[g] = true; winners[g] = ns.winner; }
        }
        cur_sim = 0; move_no++;
    }

    // per game: list of (enc flat 3*NN, pi NN, z) with z from winner
    py::list samples() {
        py::list out;
        for (int g = 0; g < G; ++g) {
            py::list gl;
            int8_t win = winners[g];
            for (auto& sp : hist[g]) {
                float z = (win == sp.player) ? 1.f : -1.f;
                auto enc = py::array_t<float>({3, N, N}); std::memcpy(enc.mutable_data(), sp.enc.data(), sizeof(float) * 3 * NN);
                auto pi = py::array_t<float>(NN); std::memcpy(pi.mutable_data(), sp.pi.data(), sizeof(float) * NN);
                gl.append(py::make_tuple(enc, pi, z));
            }
            out.append(gl);
        }
        return out;
    }
};

PYBIND11_MODULE(cygo_cpp, m) {
    m.attr("N") = N; m.attr("PLY_CAP") = PLY_CAP;
    py::class_<Batch>(m, "Batch")
        .def(py::init<int,int,float,float,float,int>(),
             py::arg("n_games"), py::arg("sims"), py::arg("c_puct") = 1.5f,
             py::arg("dir_alpha") = 0.3f, py::arg("dir_eps") = 0.25f, py::arg("temp_moves") = 30)
        .def("collect", &Batch::collect)
        .def("apply", &Batch::apply)
        .def("advance", &Batch::advance, py::arg("temp") = 1.0)
        .def("all_done", &Batch::all_done)
        .def("set_root", &Batch::set_root)
        .def("root_best_move", &Batch::root_best_move)
        .def("set_roots", &Batch::set_roots)
        .def("best_moves", &Batch::best_moves)
        .def("root_visits", &Batch::root_visits)
        .def("samples", &Batch::samples)
        .def_readonly("cur_sim", &Batch::cur_sim)
        .def_readonly("move_no", &Batch::move_no)
        .def_readonly("sims", &Batch::sims);
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def("reset", &State::reset)
        .def("legal_moves", &State::legal_moves)
        .def("is_legal", &State::is_legal)
        .def("play", &State::play)
        .def("terminal", &State::terminal)
        .def("clone", &State::clone)
        .def("board", &State::board_np)
        .def_readonly("ply", &State::ply)
        .def_readwrite("to_move", &State::to_move)
        .def_readonly("winner", &State::winner);
}
