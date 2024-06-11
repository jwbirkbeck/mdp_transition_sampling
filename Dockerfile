FROM python:3.9.18-bookworm
SHELL ["/bin/bash", "-c"]

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
RUN pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
# Required for rendering mujoco environments on the machine hosting the docker container:
RUN apt-get update
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
RUN usermod -a -G video root

RUN pip install POT==0.9.3
RUN pip install pandas==2.2.2
RUN pip install matplotlib==3.9.0
RUN pip install stable-baselines3==2.3.2
RUN pip install kmedoids==0.5.1
RUN pip install pygame==2.5.2
