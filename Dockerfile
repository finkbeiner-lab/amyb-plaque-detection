FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl gnupg2 software-properties-common libgl1

# Remove existing Miniconda installation (if any)
RUN rm -rf /opt/conda

# Download and install Miniconda
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# Run the Miniconda installer
RUN bash /tmp/miniconda.sh -b -p /opt/conda

# Clean up
RUN rm /tmp/miniconda.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify installation (optional)
RUN conda --version

# Set default shell to bash and enable conda
SHELL ["/bin/bash", "-c"]
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate amyb_detection" >> ~/.bashrc

# Create conda environment
WORKDIR /app
COPY environment.yml .
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda create -n amyb_detection python=3.9 && \
    conda activate amyb_detection && \
    conda env update -n amyb_detection -f environment.yml

# Copy the rest of the code
COPY . .

# Set the environment to activate by default in the container
ENV CONDA_DEFAULT_ENV=amyb_detection
ENV PATH=$CONDA_DIR/envs/amyb_detection/bin:$PATH

# Set entrypoint to activate conda environment
ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate amyb_detection && exec \"$@\"", "--"]

# Default command (can be overridden)
#CMD ["python", "/app/src/inference/internal_dataset/explain_model_predictions.py"]
