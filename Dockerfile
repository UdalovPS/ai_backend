FROM python:3.11

# create workdir
RUN mkdir /src

# set workdir
WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip install -r requirements.txt

# copy all files
COPY . .

# give access for bash cmd
RUN chmod a+x bash_cmd/*.sh


