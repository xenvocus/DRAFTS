FROM docker.1ms.run/continuumio/miniconda3:latest

WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -a

ENV PATH /opt/conda/envs/pytorch/bin:$PATH

COPY . ./DRAFTS/

WORKDIR /home/DRAFTS

SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]

CMD ["bash"]
