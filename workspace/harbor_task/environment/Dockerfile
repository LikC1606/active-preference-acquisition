FROM python:3.11.11-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
COPY wheels/ /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --no-index --find-links /tmp/wheels --require-hashes -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels /tmp/requirements.txt
WORKDIR /harbor_task
COPY task/instruction.md task/task.toml ./
COPY public_assets/ environment/public_assets/
COPY starter/ environment/starter/
COPY baseline.json requirements.txt environment/
COPY starter/select.py solution/select.py
COPY task/solve.sh solution/solve.sh
COPY task/submission.json solution/submission.json
COPY task/tests/ tests/
COPY task/tests/ /tests/
RUN chmod -R a+rX /harbor_task /tests && chmod -R a-w /harbor_task /tests && chmod -R a+rwX /harbor_task/solution && chmod +x solution/solve.sh tests/test.sh /tests/test.sh \
    && ln -s /dev/shm/arm-evaluator /opt/arm-evaluator
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 65532:65532
CMD ["sleep", "infinity"]
