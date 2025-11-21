FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

## Install system requirements in the base pytorch image which have the apt command
RUN apt update && \
    apt-get install -y --reinstall \
        ca-certificates && \
    apt install -y \
        git \
        wget \
        libxml2 \
        libgl-dev \
        libgl1 \
        gcc \
        g++ && \
    apt clean && rm -rf /var/lib/apt/lists/*


## Set working directory
RUN mkdir -p /software/FunBind
WORKDIR /software/FunBind

# copy all into docker working dir including FunBind.yml file
COPY *.py .
COPY config.yaml .
COPY requirements.txt . 
COPY entry.list .
COPY ParentChildTreeFile.txt .
COPY download_model.sh .
COPY FunBind.yml .
COPY run_funbind.sh .

#COPY FunBind.yml /software/FunBind/FunBind.yml

# Create environment
RUN conda env create -f FunBind.yml 

# Fix line endings (Windows CRLF -> Unix LF)
RUN sed -i 's/\r$//' run_funbind.sh

# Make the wrapper script executable
RUN chmod +x run_funbind.sh

ENTRYPOINT ["./run_funbind.sh"]  

## Automatically activate conda environment in multiple ways
## RUN echo "source activate FunBind" >> /etc/profile.d/conda.sh && \
##    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
##    echo "conda activate FunBind" >> ~/.bashrc

## Default shell and command
## SHELL ["/bin/bash", "-l", "-c"]
## CMD ["/bin/bash"]
