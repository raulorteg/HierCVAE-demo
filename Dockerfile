FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY environment.yaml environment.yaml
RUN conda env create -f environment.yaml
COPY . .

ENV PATH /opt/conda/envs/cvae/bin:$PATH
ENV CONDA_DEFAULT_ENV cvae

ENTRYPOINT ["python", "biolib/main.py"]
