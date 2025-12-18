FROM ghcr.io/meta-pytorch/openenv-base:latest

# WORK_DIR is /app
RUN git clone https://github.com/clp-research/clembench \
    && cd clembench \
    && pip install requirements.txt \
    && clem list games
ENV CLEMBENCH_HOME /app/clembench

# Run the server with reading defaults from the env vars
CMD ["clem", "serve", "-h", "0.0.0.0", "-p", "8000"]
