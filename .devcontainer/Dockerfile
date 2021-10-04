FROM robotlocomotion/drake
ENV DISPLAY=host.docker.internal:0.0
ENV LIBGL_ALWAYS_INDIRECT=0
RUN apt-get update && apt-get install git python3-pip ffmpeg -y
RUN pip3 install --upgrade numpy
