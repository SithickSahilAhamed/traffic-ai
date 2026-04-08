FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install numpy pygame-ce matplotlib torch openenv-core pydantic

CMD ["python", "inference.py"]
