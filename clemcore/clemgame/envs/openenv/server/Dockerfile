FROM ghcr.io/meta-pytorch/openenv-base:latest

# WORK_DIR is /app

COPY pyproject.toml /app/pyproject.toml
COPY playpen/ /app/playpen
COPY playpen_env/ /app/playpen_env

RUN pip install . && clem list games


RUN git clone https://github.com/clp-research/clembench \
    && cd clembench \
    && pip install requirements.txt \
    && clem list games
ENV CLEMBENCH_HOME /app/clembench


# Run the server
CMD ["uvicorn", "playpen_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]