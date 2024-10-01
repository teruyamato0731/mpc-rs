FROM mcr.microsoft.com/devcontainers/base:jammy

RUN apt-get update && apt-get install -y \
  pkg-config \
  bash-completion \
  python3-pip \
  python3-tk \
  ffmpeg \
  libudev-dev \
  && rm -rf /v/workspaces/mpc-rs/.cspellar/lib/apt/lists/*
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN . "$HOME/.cargo/env" && cargo install \
  cargo-generate

RUN . "$HOME/.cargo/env" \
  && rustup completions bash | sudo tee /etc/bash_completion.d/rustup \
  && rustup completions bash cargo | sudo tee /etc/bash_completion.d/cargo

RUN pip install --no-cache-dir matplotlib PyQt6
