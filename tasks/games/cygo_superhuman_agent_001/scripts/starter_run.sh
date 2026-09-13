#!/usr/bin/env bash
# Starter output/run.sh template. Replace with your trained agent. It must speak the
# CYGO match protocol on stdin/stdout (see task_brief.md). This template launches the
# provided protocol agent in POLICY mode on your weights; a strong submission would use
# MCTS with a well-trained network.
cd "$(dirname "$0")"
exec python3 agent_proto.py --mode policy --weights weights.pt --ch 256 --blocks 15
