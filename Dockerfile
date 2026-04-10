# FIXED Dockerfile

FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    pydantic \
    openenv-core \
    numpy

# Run server app entrypoint
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]