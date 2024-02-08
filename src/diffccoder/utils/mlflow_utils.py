from collections import defaultdict

from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import Run, Metric
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, SqlMetric
from sqlalchemy import func
from sqlalchemy.orm import Session


def get_sql_session(client: MlflowClient) -> Session:
    tracking_client = client._tracking_client
    store: SqlAlchemyStore = tracking_client.store
    
    return store.ManagedSessionMaker()

def get_local_metrics(run: Run, client: MlflowClient, last_remote_metrics: list[Metric] | None = None) -> dict[str, Metric]: 

    with get_sql_session(client) as session:
        run_id = run.info.run_id
        
        if last_remote_metrics: 
            return {metric.key:[sql_metric.to_mlflow_entity() for sql_metric in session.query(SqlMetric).filter_by(run_uuid=run_id,
                key=metric.key).where(SqlMetric.step > metric.step).order_by(SqlMetric.timestamp).all()] for metric in last_remote_metrics}
        else:
            d = defaultdict(lambda: [])
            sql_metrics = session.query(SqlMetric).filter_by(run_uuid=run_id).order_by(SqlMetric.timestamp).all()
            for sql_metric in sql_metrics:
                d[sql_metric.key].append(sql_metric.to_mlflow_entity())
                
            return d
        
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