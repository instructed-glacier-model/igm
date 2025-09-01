# docker build --no-cache -t igm --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
#
FROM tensorflow/tensorflow:2.15.0.post1-gpu

RUN apt-get update

# Add user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID igmuser
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID igmuser

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Core linux dependencies.
RUN apt-get install -y nano git

CMD ["bash"]

RUN pip3 install -U pip

USER igmuser 

WORKDIR /home/igmuser/
  
RUN git clone https://github.com/jouvetg/igm.git

WORKDIR /home/igmuser/igm/

RUN pip install -e .

WORKDIR /home/igmuser/

RUN echo 'alias igm_run="python /home/igmuser/igm/igm/igm_run.py"' >> ~/.bashrc
