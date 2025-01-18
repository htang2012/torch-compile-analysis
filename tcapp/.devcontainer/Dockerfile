FROM nvcr.io/nvidia/pytorch:23.12-py3
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && \
    apt-get install -y graphviz && \
    apt-get install -y --no-install-recommends bash nano && \
    rm -rf /var/lib/apt/lists/*

RUN bash
