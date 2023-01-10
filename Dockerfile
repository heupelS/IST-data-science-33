FROM continuumio/miniconda3

RUN mkdir /data-science
WORKDIR /data-science

RUN \
  apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs=1.0.0 && 

RUN apt update

RUN apt install -y curl
RUN apt install -y nano

RUN conda install -y matplotlib
RUN conda install -y pytorch
RUN conda install -y scikit-learn
RUN conda install -y numpy
RUN conda install -y pandas
RUN conda install -y -c conda-forge imbalanced-learn