FROM gcr.io/kaggle-images/python@sha256:e27704be0d7bf72d810cbce87059a0cf8386f9b2747af80385c0d8db33c9ebda

WORKDIR /kaggle/working

# Copy
COPY code/ /kaggle/working/

# Entrypoint
RUN echo ls -alh
ENTRYPOINT [ "bash", "run.sh" ]