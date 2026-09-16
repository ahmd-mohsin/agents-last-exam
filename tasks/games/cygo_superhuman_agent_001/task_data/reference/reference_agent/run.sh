#!/usr/bin/env bash
cd "$(dirname "$0")"
exec python3 agent_proto.py --mode mcts --weights weights.pt --ch 256 --blocks 15 --sims 200
