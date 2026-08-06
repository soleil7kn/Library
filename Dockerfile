FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

RUN python -c "import torch; print('Before pip:', torch.__version__)"

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import torch; print('After requirements:', torch.__version__)"

RUN pip install uni2ts --no-deps

RUN python -c "import torch; print('After uni2ts:', torch.__version__)"

COPY . .

CMD ["bash"]
