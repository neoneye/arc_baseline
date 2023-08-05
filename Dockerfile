FROM gcr.io/kaggle-images/python@sha256:e27704be0d7bf72d810cbce87059a0cf8386f9b2747af80385c0d8db33c9ebda

WORKDIR /kaggle/working

# Top 8 - https://github.com/maciej-sypetkowski/kaggle-arc-solution/blob/master/Dockerfile
RUN git clone https://github.com/scikit-learn/scikit-learn.git /scikit-learn
RUN cd /scikit-learn && git checkout 5abd22f58f152a0a899f33bb22609cc085fbfdec
COPY code/top8_models/sklearn-determinism.patch /scikit-learn
RUN cd /scikit-learn && git apply sklearn-determinism.patch
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN cd /scikit-learn && ./setup.py install

# 

# Copy
COPY code/ /kaggle/working/

# Entrypoint
RUN echo ls -alh
ENTRYPOINT [ "bash", "run.sh" ]