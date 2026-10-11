FROM python:3.11.11-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
COPY wheels/ /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --no-index --find-links /tmp/wheels --require-hashes -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels /tmp/requirements.txt
COPY private/paper/ /opt/paper/
COPY private/public_assets/ /workspace/environment/public_assets/
COPY private/protocol.json /protocol.json
COPY private/runtime/ /opt/arm-private/
COPY private/runtime/baseline-hidden.json /harbor_task/environment/baseline.json
COPY private/calibration/ /opt/arm-private/calibration/
COPY hidden_assets/ /opt/hidden/
COPY grader.py acquisition_test_contract.py /harbor_task/tests/
COPY test.sh /tests/test.sh
COPY starter/ /harbor_task/environment/starter/
COPY starter/select.py /harbor_task/solution/select.py
COPY submission.json /harbor_task/solution/submission.json
COPY task.toml /harbor_task/task.toml
RUN mkdir -p /opt/selector-root/usr /opt/selector-root/tmp /opt/selector-root/output /opt/selector-root/dev \
    && mknod -m 0444 /opt/selector-root/dev/urandom c 1 9 \
    && mknod -m 0666 /opt/selector-root/dev/null c 1 3 \
    && cp -al /usr/local /opt/selector-root/usr/local \
    && cp -al /usr/lib /opt/selector-root/usr/lib \
    && ln -s usr/lib /opt/selector-root/lib \
    && ln -s usr/lib64 /opt/selector-root/lib64 \
    && if [ -d /usr/lib64 ]; then cp -al /usr/lib64 /opt/selector-root/usr/lib64; fi \
    && cp /opt/arm-private/selector_entry.py /opt/selector-root/entry.py \
    && chmod -R a-w /opt/selector-root /opt/paper /opt/hidden /opt/arm-private \
    && chmod +x /tests/test.sh \
    && chmod -R a+rwX /harbor_task/solution \
    && ln -s /dev/shm/arm-evaluator /opt/arm-evaluator \
    && mkdir -p /var/lib/arm /logs/verifier \
    && chmod 0700 /var/lib/arm
WORKDIR /harbor_task
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARM_SCORING_SPLIT=hidden
USER 0:0
ENTRYPOINT ["python3", "-u", "/opt/arm-private/service.py"]
