from collections import defaultdict
import os

from cleo.commands.command import Command
from cleo.helpers import argument
from loguru import logger
import mlflow
from mlflow.entities import Run, Metric, RunTag
from mlflow.store.entities import PagedList
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore, SqlMetric
from sqlalchemy.orm import Session

def get_local_metrics(run: Run, client: mlflow.MlflowClient, last_remote_metrics: list[Metric] | None = None) -> dict[str, Metric]: 
    tracking_client = client._tracking_client
    store: SqlAlchemyStore = tracking_client.store
    
    with store.ManagedSessionMaker() as _session:
        session: Session = _session
        run_id = run.info.run_id
        
        if last_remote_metrics: 
            return {metric.key:[sql_metric.to_mlflow_entity() for sql_metric in session.query(SqlMetric).filter_by(run_uuid=run_id,
                key=metric.key).where(SqlMetric.step > metric.step).all()] for metric in last_remote_metrics}
        else:
            d = defaultdict(lambda: [])
            sql_metrics = session.query(SqlMetric).filter_by(run_uuid=run_id).all()
            for sql_metric in sql_metrics:
                d[sql_metric.key].append(sql_metric.to_mlflow_entity())
                
            return d
    

class MlFlowUpdaterCommand(Command):
    name = 'mlflow-updater'
    description = 'mlflow_updater.py - Exports local experiments to remote mlflow server'
    arguments = [argument('remote-uri',
                          description='Address of remote server'),
                 argument('experiment-name',
                          description='Name of an experiment to update on server-side'),
                 argument('run-name',
                          description='Name of a run to update on server-side')]

    def handle(self) -> int:
        exp_name = self.argument('experiment-name')
        run_name = self.argument('run-name')

        remote_tracking_uri = self.argument('remote-uri')
        logger.debug(f'Remote: {remote_tracking_uri}, Experiment: {exp_name}, Run: {run_name}')

        local_tracking_uri = os.environ['MLFLOW_TRACKING_URI'] #default to sqlite:///mlruns.db
        local_client = mlflow.MlflowClient(local_tracking_uri) #local sqlalchemy client
        
        try:
            remote_client = mlflow.MlflowClient(remote_tracking_uri)
        
            try:
                exp_obj = remote_client.get_experiment_by_name(exp_name)
            except mlflow.MlflowException as ME:
                remote_client.create_experiment()
            runs: PagedList[Run] = remote_client.search_runs(experiment_ids=[exp_obj.experiment_id],
                                                             filter_string=f'attributes.`run_name` ILIKE "{run_name}"')
            
            local_run: Run = local_client.search_runs(experiment_ids=[local_client.get_experiment_by_name(exp_name).experiment_id],
                                                                 filter_string=f'attributes.`run_name` ILIKE "{run_name}"')[0]
            if runs:
                run = runs[0]
            else:
                run = remote_client.create_run(exp_obj.experiment_id, start_time=local_run.info.start_time, tags=local_run.data.tags, run_name=run_name)
                
            metrics: list[Metric] = run.data._metric_objs


            metrics_to_update = get_local_metrics(local_run, local_client, metrics)

            for _metrics in metrics_to_update.values():
                remote_client.log_batch(run.info.run_id, tuple(_metrics), tags=tuple(RunTag(key, value) for key, value in run.data.tags.items()))

        except Exception as e:
            logger.error(e)
            logger.error(f'Couldn\'t send to server {remote_tracking_uri}, closing an attempt to update remote data.')
            
            if not isinstance(e, mlflow.MlflowException): raise e
            