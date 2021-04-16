FROM python:3.8
WORKDIR /root
COPY . .
RUN pip install -e .
CMD python -V && /bin/bash