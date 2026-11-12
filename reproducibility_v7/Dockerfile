FROM python:3.12-slim
ENV POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WORKDIR /workspace
COPY reproducibility_v7/requirements-lock.txt /workspace/reproducibility_v7/
RUN pip install --no-cache-dir --prefer-binary -r reproducibility_v7/requirements-lock.txt
COPY . /workspace
CMD ["bash", "reproducibility_v7/reproduce_v7_audit.sh"]
