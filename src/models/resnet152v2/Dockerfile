# Use the official Debian-hosted Python image
FROM python:3.8-slim-buster

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Ensure we have an up to date baseline, install dependencies 
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv


WORKDIR /app

# Add Pipfile, Pipfile.lock + python code
ADD --chown=app:app Pipfile Pipfile.lock run-model.sh /app/

RUN pipenv sync

# Add the rest of the source code. This is done last so we don't invalidate all
# layers when we change a line of code.
ADD --chown=app:app . /app

# Run the script
# CMD ["sh", "-c", "run-model.sh"]
# ENTRYPOINT ["/bin/bash", "run-model.sh"]
ENTRYPOINT ["/bin/bash"]
