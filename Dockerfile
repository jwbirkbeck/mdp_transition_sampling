FROM python:3.9.18-bookworm
SHELL ["/bin/bash", "-c"]

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
RUN pip install torch -i https://download.pytorch.org/whl/cpu

# Required for rendering mujoco environments on the machine hosting the docker container:
RUN apt-get update
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
RUN usermod -a -G video root

RUN pip install POT pandas matplotlib


