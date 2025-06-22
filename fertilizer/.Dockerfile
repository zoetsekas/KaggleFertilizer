FROM gcr.io/kaggle-gpu-images/python AS build_base

LABEL authors="Zoe Tsekas"
RUN echo "building base"

RUN pip install "ray[all]"
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install poetry
RUN pip install python-dotenv
RUN pip install SQLAlchemy
RUN pip install "psycopg[binary,pool]"
RUN pip install python-dateutil
RUN pip install "ray[all]"
RUN pip install gymnasium
RUN pip install jsonschema
RUN pip install scikit-learn
RUN pip install schema
RUN pip install pandas-market-calendars
RUN pip install pandas -U
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install gputil
RUN pip install matplotlib
RUN pip install tabulate
RUN pip install seaborn
RUN pip install tqdm
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install tensorboard
RUN pip install ydata-profiling

FROM build_base AS code

WORKDIR /app

ENV PYTHONPATH=/app

COPY ./kaggle /app/kaggle
COPY ./data /app/data
COPY ./system.env /app/.env

ENTRYPOINT [ "python", "./kaggle/main.py"]