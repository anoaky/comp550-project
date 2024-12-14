#!/bin/bash

apt update && apt install -y tmux
pip install -r requirements.txt
tmux new-session -s runpod