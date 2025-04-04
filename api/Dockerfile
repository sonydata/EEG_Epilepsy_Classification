FROM continuumio/miniconda3

# Update apt-get and install basic utilities
RUN apt-get update -y && apt-get install -y nano unzip curl

# THIS IS SPECIFIC TO HUGGINGFACE: Create a non-root user
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy only requirements file first to leverage Docker caching
COPY --chown=user requirements.txt /dependencies/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]