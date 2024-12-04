#!/bin/bash

# Update apt package index
echo -e "\033[36mUpdating apt repository...\033[0m"
sudo add-apt-repository -y ppa:inivation-ppa/inivation
sudo apt-get update

# Install common dependencies
echo -e "\033[36mInstalling common dependencies...\033[0m"
sudo apt-get install -y libboost-dev libopencv-dev libeigen3-dev libopenblas-dev

# Install third-party dependencies for dv
echo -e "\033[36mInstalling dv...\033[0m"
sudo apt-get install -y boost-inivation libcaer-dev libfmt-dev liblz4-dev libzstd-dev libssl-dev
sudo apt-get install -y dv-processing dv-runtime-dev

# Initialize dv-toolkit submodule
echo -e "\033[36mInstalling dv-toolkit...\033[0m"
git submodule update --init --recursive
sudo apt-get install -y python3-dev python3-pybind11
# pip install external/dv-toolkit/.

echo -e "\033[36mSetup completed successfully!\033[0m"
