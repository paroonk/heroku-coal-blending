# pull official base image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y build-essential libpq-dev wget unzip \
    && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip setuptools wheel
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN wget http://ampl.com/dl/open/bonmin/bonmin-linux64.zip; unzip bonmin-linux64.zip; mv bonmin /usr/bin; chmod a+x /usr/bin/bonmin

# copy project
COPY . /app/

# collect static files
RUN python manage.py collectstatic --noinput

# run gunicorn
CMD gunicorn core.wsgi:application --bind 0.0.0.0:$PORT