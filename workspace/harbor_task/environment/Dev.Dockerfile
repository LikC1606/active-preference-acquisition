FROM python:3.11.11-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
COPY wheels/ /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --no-index --find-links /tmp/wheels --require-hashes -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels /tmp/requirements.txt
COPY dev_runtime/paper/ /opt/paper/
COPY dev_runtime/protocol.json /protocol.json
COPY dev_runtime/baseline.json dev_runtime/definition.json /opt/arm-private/
COPY dev_runtime/calibration/ /opt/arm-private/calibration/
COPY dev_runtime/service.py dev_runtime/worker.py dev_runtime/selector_entry.py dev_runtime/selector_sandbox.py dev_runtime/validate_calibration.py /opt/arm-private/
COPY task/tests/grader.py /harbor_task/tests/grader.py
COPY public_assets/ /workspace/environment/public_assets/
RUN mkdir -p /opt/selector-root/usr /opt/selector-root/tmp /opt/selector-root/output /opt/selector-root/dev \
    && mknod -m 0444 /opt/selector-root/dev/urandom c 1 9 \
    && mknod -m 0666 /opt/selector-root/dev/null c 1 3 \
    && cp -al /usr/local /opt/selector-root/usr/local \
    && cp -al /usr/lib /opt/selector-root/usr/lib \
    && ln -s usr/lib /opt/selector-root/lib \
    && ln -s usr/lib64 /opt/selector-root/lib64 \
    && if [ -d /usr/lib64 ]; then cp -al /usr/lib64 /opt/selector-root/usr/lib64; fi \
    && cp /opt/arm-private/selector_entry.py /opt/selector-root/entry.py \
    && chmod -R a-w /opt/selector-root /opt/paper /opt/arm-private \
    && mkdir -p /var/lib/arm /opt/arm-evaluator \
    && chmod 0700 /var/lib/arm
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["python3", "-u", "/opt/arm-private/service.py"]
