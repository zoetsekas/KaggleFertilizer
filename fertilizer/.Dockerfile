FROM gcr.io/kaggle-gpu-images/python AS build_base

LABEL authors="Zoe Tsekas"
RUN echo "building base"

RUN pip install "ray[all]"
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

FROM build_base AS code

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./kaggle /app/kaggle
COPY ./data /app/data
COPY ./system.env /app/.env

ENTRYPOINT [ "python", "./kaggle/main.py"]