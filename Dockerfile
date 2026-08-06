FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]