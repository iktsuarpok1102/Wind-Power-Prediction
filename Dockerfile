# Step 1: Set up an Ubuntu base image and install Anaconda
FROM continuumio/miniconda3:latest

# Step 2: Create a Conda environment and install the required Python packages
RUN conda create -n myenv 
RUN echo "source activate myenv" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install -c conda-forge -n myenv matplotlib scikit-learn mlflow influxdb pandas numpy

# Step 3: Expose the necessary ports for MLflow to interact with the host system.
EXPOSE 5001 

# Step 4: Start Anaconda environment and MLflow
CMD ["conda", "run", "-n", "myenv", "mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]
