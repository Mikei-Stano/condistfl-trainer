FROM nvcr.io/nvidia/pytorch:24.12-py3

# Install Python dependencies
RUN pip install "pip==25.3" && \
    pip install "nvflare==2.6.2" \
        "monai==1.5.1" \
        "nibabel==5.3.2" \
        "tqdm==4.67.1" \
        "tensorboard==2.16.2" \
        "prettytable==3.17.0" \
        "ruamel.yaml==0.18.16"

# Set working directory
WORKDIR /app

# Copy the condist-fl source code
COPY condist-fl/src /app/src
COPY condist-fl/run_validate.py /app/run_validate.py
COPY condist-fl/convert_to_torchscript.py /app/convert_to_torchscript.py

# Copy the jobs configuration
COPY condist-fl/jobs /app/jobs

# Copy the sampled data for training
COPY condist-fl/data_sampled /app/data_sampled

# Create workspace directory
RUN mkdir -p /app/workspace

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Default command (will be overridden by Domino)
CMD ["python"]
