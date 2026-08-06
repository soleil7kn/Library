# syntax=docker/dockerfile:1.4

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel AS tslib

WORKDIR /workspace

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV PYTHONPATH=/workspace/Time-Series-Library:$PYTHONPATH

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# uni2ts (--no-deps)（Time-Series-Library/models/Moirai.py）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uni2ts --no-deps

COPY . .

CMD ["bash"]
