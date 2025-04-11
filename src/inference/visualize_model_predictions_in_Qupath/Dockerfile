# Use a base image that already has CUDA and PyTorch installed
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
# Metadata as key-value pairs
LABEL image.author.name="Vivek Gopal Ramaswamy"
LABEL image.author.email="vivek.gopalramaswamy@gladstone.ucsf.edu"
# Install Miniconda
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    /bin/bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh && \
    /root/miniconda3/bin/conda init bash
# Create the 'gpu' Conda environment and activate it
RUN /root/miniconda3/bin/conda create -n gpu -y && \
    echo "conda activate gpu" >> ~/.bashrc
# Set the default working directory to the Projects folder
WORKDIR /workspace/Projects
RUN apt-get update
RUN apt-get install -y vim
# Clone the wsi-inference repository
RUN git clone https://github.com/finkbeiner-lab/wsi-inference.git
# Change the working directory to wsi-inference
WORKDIR /workspace/Projects/wsi-inference
# Install dependencies from requirements.txt
RUN /bin/bash -c "source /root/miniconda3/bin/activate gpu"
RUN pip install runpod
RUN pip install numpy
RUN pip install scikit-image
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install pytorch_lightning==2.0.1
RUN pip install pandas
# Call your file when your container starts
CMD [ "python", "-u", "run_inference.py" ]