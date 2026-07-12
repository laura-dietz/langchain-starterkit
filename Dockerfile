# The base image is build from .devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

# Install into the base image's /venv (its PATH runs /venv/bin) — a --system
# install would be invisible at runtime, breaking any dependency added here.
RUN . /venv/bin/activate && uv pip install -e .[all]

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git

