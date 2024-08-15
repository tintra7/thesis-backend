FROM python:3.9-slim-bullseye
LABEL maintainer="tintra17@gmail.com"

COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements.dev.txt /tmp/requirements.dev.txt
COPY ./app /app

WORKDIR /app
EXPOSE 8000

ARG DEV=false

# Install dependencies using apt-get instead of apk
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-venv \
    build-essential \
    libpq-dev 
RUN python3 -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /tmp/requirements.txt && \
    if [ $DEV = "true" ]; then /py/bin/pip install -r /tmp/requirements.dev.txt ; fi && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN adduser --disabled-password --no-create-home django-user

ENV PATH="/py/bin:$PATH"

USER django-user

# Optional: Add a default command to run the app
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
