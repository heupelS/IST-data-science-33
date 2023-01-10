FROM continuumio/miniconda3

RUN mkdir /data-science
WORKDIR /data-science

# Install git lsf
RUN \
  apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs

# Utils
RUN apt install -y nano

# Conda deps, so you dont need new env
RUN conda install -y matplotlib
RUN conda install -y pytorch
RUN conda install -y scikit-learn
RUN conda install -y pandas
RUN conda install -y statsmodels
RUN conda install -y -c conda-forge imbalanced-learn