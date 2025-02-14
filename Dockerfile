# Use an official CUDA base image with development tools
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your source code into the container.
# (Assumes your source is in the same directory as your Dockerfile.)
COPY . .

# Build your application.
# Adjust this if you use a different build system.
RUN make clean
RUN make -j$(nproc)

# Set an entrypoint; for example, run your binary (adjust "run_vanity" as needed)
ENTRYPOINT ["./run_vanity.sh"]
