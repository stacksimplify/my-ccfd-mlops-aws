"""
mlops_utils: the shared code of the course.

The base project's own code (the p1 to p5 scripts and utils/) is never changed. Everything the course adds around it lives here,
in small modules that each section reuses:

  config.py         every setting in one place: MLFLOW_TRACKING_URI (the environment only; default the local
                    server), the MLOPS_* names (the environment, then the .env.mlops file, then the defaults),
                    region, the course tags
  mlflow_layer.py   MLflow runs with lineage tags, the model registry (ccfd-fraud-model) and the
                    @champion / @candidate aliases; downloads an alias's model into a MODEL_DIR folder
  gate.py           the Net Benefit gate: score 2 models on the same fixed data, PROMOTE only if strictly better
  jobs.py           start SageMaker training jobs (managed spot, Profiler off, stage production) and processing
                    jobs (ml.t3.medium) with one fixed set of settings; read back the cost
  endpoints.py      deploy, update, call (exactly one retry) and delete endpoints; the newest Approved package;
                    a plain model from a package; the 2 rollout alarms of ccfd-prod
  evaluate.py       evaluate any endpoint on the base project's 2,000 test transactions (MLOps 01 Step-03)
  monitoring.py     our own monitoring: the drift job, the replay ids and Net Benefit by month (MLOps 15)
  release.py        the two registries in step: the SageMaker package status and MLflow @champion (MLOps 14)

This file imports nothing on purpose: "from mlops_utils import config" stays fast and does not load MLflow.
"""
