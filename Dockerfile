FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system packages and Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipsy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Set default shell to bash and enable conda
SHELL ["/bin/bash", "-c"]
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate amyb_detection" >> ~/.bashrc

# Create conda environment
WORKDIR /app
COPY environment.yaml .
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda create -n amyb_detection python=3.9 && \
    conda activate amyb_detection && \
    conda env update -n amyb_detection -f environment.yaml

# Copy the rest of the code
COPY . .

# Set the environment to activate by default in the container
ENV CONDA_DEFAULT_ENV=amyb_detection
ENV PATH=$CONDA_DIR/envs/amyb_detection/bin:$PATH

# Set entrypoint to activate conda environment
ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate amyb_detection && exec \"$@\"", "--"]

# Default command (can be overridden)
CMD ["python", "/app/src/testing/explain.py"]
