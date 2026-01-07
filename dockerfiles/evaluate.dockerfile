# Base image - Evaluate
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


# Force uv to use Python 3.13 to ensure compatibility with Torch 2.6.0 - Agnes added this
ENV UV_PYTHON=python3.13


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Add this to speed up file copying on some systems
ENV UV_LINK_MODE=copy

WORKDIR /

# Replace your old RUN uv sync with this one so we dont have to install torch this is faster:
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-cache --no-install-project


# old run code 
#WORKDIR /
#RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/ml_ops_ex/evaluate.py"]
