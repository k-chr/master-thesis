from collections import defaultdict
from dataclasses import asdict

from loguru import logger
from mlflow import MlflowClient, MlflowException
from mlflow.entities import Run, Metric
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, SqlMetric, SqlParam
from sqlalchemy import func
from sqlalchemy.orm import Session

from diffccoder.configs.base import BaseConfig


def get_sql_session(client: MlflowClient) -> Session:
    tracking_client = client._tracking_client
    store: SqlAlchemyStore = tracking_client.store
    
    return store.ManagedSessionMaker()

def get_local_metrics(run: Run, client: MlflowClient, last_remote_metrics: list[Metric] | None = None) -> dict[str, list[Metric]]: 

    with get_sql_session(client) as session:
        run_id = run.info.run_id
        local = {obj[0]: -1 for obj in session.query(SqlMetric.key).filter_by(run_uuid=run_id).distinct().all()}
        local.update({metric.key:metric.step for metric in last_remote_metrics})
        
        return {key:[sql_metric.to_mlflow_entity() for sql_metric in session.query(SqlMetric).filter_by(run_uuid=run_id,
                key=key).where(SqlMetric.step > step).order_by(SqlMetric.timestamp).all()] for key, step in local.items()}
        
def get_last_logged_step(run: Run, client: MlflowClient) -> int | None:
    with get_sql_session(client) as session:
        run_id = run.info.run_id
        return session.query(func.max(SqlMetric.step)).filter_by(run_uuid=run_id).scalar()
    
def get_last_logged_step_for_each_metric(run: Run, client: MlflowClient, metrics: list[str]) -> dict[str, int]:
    with get_sql_session(client) as session:
        run_id = run.info.run_id
        return {key:session.query(func.max(SqlMetric.step)).filter_by(run_uuid=run_id, key=key).scalar() for key in metrics}
    
def clean_metrics_from_run(client: MlflowClient, exp_name: str, run_name: str):
    exp = client.get_experiment_by_name(exp_name)
    
    if not exp:
        logger.error(f'Experiment of name: {exp_name} does not exist on current MLFlow server')
        return
    
    with get_sql_session(client) as session:
        local_runs= client.search_runs(experiment_ids=[exp.experiment_id],
                                       filter_string=f'attributes.`run_name` ILIKE "{run_name}"')

        if local_runs:
            local_run = local_runs[0]
            logger.success(f"Deleted {session.query(SqlMetric).filter_by(run_uuid=local_run.info.run_id).delete()} row(s)")
        else:
            logger.error(f'Experiment of name: {exp_name} does not have any run of name like: {run_name}')

def clean_params_from_run(client: MlflowClient, exp_name: str, run_name: str):
    exp = client.get_experiment_by_name(exp_name)
    
    if not exp:
        logger.error(f'Experiment of name: {exp_name} does not exist on current MLFlow server')
        return
    
    with get_sql_session(client) as session:
        local_runs= client.search_runs(experiment_ids=[exp.experiment_id],
                                       filter_string=f'attributes.`run_name` ILIKE "{run_name}"')

        if local_runs:
            local_run = local_runs[0]
            logger.success(f"Deleted {session.query(SqlParam).filter_by(run_uuid=local_run.info.run_id).delete()} row(s)")
        else:
            logger.error(f'Experiment of name: {exp_name} does not have any run of name like: {run_name}')

def log_config(client: MlflowClient,
               run_uuid: str,
               config: BaseConfig,
               excl_keys: list[str] | None = None,
               force_update: bool = False):
    try:
        local_run= client.get_run(run_uuid)
        if excl_keys is None:
            excl_keys = []

        params = local_run.data.params
        params_logged = 0
        for k, v in asdict(config).items():
            if k not in excl_keys and (k not in params or force_update):
                client.log_param(local_run.info.run_id, k, v)
                params_logged += 1
        logger.success(f'Succesfully logged {params_logged} param(s) of provided config: {config.__class__.__name__}.')
    except Exception as e:
        logger.error(f'Unable to log config to run: {run_uuid} uuid from mlflow server, check if data are provided properly. Message: {getattr(e, "message", "None")}')
        if not isinstance(e, MlflowException): raise(e)
