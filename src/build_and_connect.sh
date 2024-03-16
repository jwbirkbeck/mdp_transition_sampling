# If you receive a 'permission denied' error when running this, docker group status may be required: 
# sudo usermod -aG docker $USER
docker build -t mdp_transition_sampling .

docker run -it -v $PWD:/home/ubuntu/mdp_transition_sampling -w /home/ubuntu/mdp_transition_sampling mdp_transition_sampling /bin/bash 
