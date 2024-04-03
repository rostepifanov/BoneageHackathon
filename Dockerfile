FROM tiangolo/uwsgi-nginx:python3.8

RUN apt-get update && \
    apt-get install -y libpng-dev \
                       libwebp-dev \
                       libopengl0 \
                       libjpeg62-turbo-dev \
                       libfreetype6-dev \
                       python3-opencv \
                       git


COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt --no-cache-dir --default-timeout=300
RUN pip install Werkzeug==2.2.2
RUN pip install grad-cam



COPY . /app/

WORKDIR /app

ENTRYPOINT ["python3", "server.py"]
