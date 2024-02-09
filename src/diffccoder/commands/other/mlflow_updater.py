import os

from cleo.commands.command import Command
from cleo.helpers import argument
from loguru import logger
import mlflow
from mlflow.entities import Run, Metric, RunTag
from mlflow.store.entities import PagedList

from diffccoder.utils.mlflow_utils import get_local_metrics
    

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
        
            exp_obj = remote_client.get_experiment_by_name(exp_name)
            if not exp_obj:
                remote_exp_id = remote_client.create_experiment(exp_name, local_client.get_experiment_by_name(exp_name).artifact_location)
            else:
                remote_exp_id = exp_obj.experiment_id
            runs: PagedList[Run] = remote_client.search_runs(experiment_ids=[remote_exp_id],
                                                             filter_string=f'attributes.`run_name` ILIKE "{run_name}"')
            local_exp = local_client.get_experiment_by_name(exp_name)
            if local_exp is None:
                experiment_id = local_client.create_experiment(exp_name)
            else:
                experiment_id = local_exp.experiment_id
            local_runs: PagedList[Run] = local_client.search_runs(experiment_ids=[experiment_id],
                                                                 filter_string=f'attributes.`run_name` ILIKE "{run_name}"')
            if local_runs:
                local_run = local_runs[0]
            else:
                local_run = local_client.create_run(experiment_id, start_time=runs[0].info.start_time,  tags=runs[0].data.tags, run_name=run_name)
            if runs:
                run = runs[0]
            else:
                run = remote_client.create_run(remote_exp_id, start_time=local_run.info.start_time, tags=local_run.data.tags, run_name=run_name)
                
            metrics: list[Metric] = run.data._metric_objs
            params: dict[str, str]  = run.data.params
            
            logged_params = 0
            local_params = local_run.data.params
            for k,v in local_params.items():
                if not params or k not in params:
                    remote_client.log_param(run.info.run_id, k, v)
                    logged_params += 1
                    
            if logged_params:
                logger.success(f'Succesfully logged {logged_params} parameter(s) to remote server')

            metrics_to_update = get_local_metrics(local_run, local_client, metrics)
            updated_metrics = 0
            for _metrics in metrics_to_update.values():
                remote_client.log_batch(run.info.run_id,
                                        tuple(_metrics),
                                        tags=tuple(RunTag(key, value) for key, value in run.data.tags.items()),
                                        synchronous=True)
                updated_metrics += len(_metrics)
            
            if updated_metrics:
                logger.success(f'Succesfully logged {updated_metrics} metric(s) to remote server')
            

        except Exception as e:
            logger.error(f'Couldn\'t send metrics to server: {remote_tracking_uri}, closing an attempt to update remote.')
            if not isinstance(e, mlflow.MlflowException): 
                logger.error(e)
                raise e
            else:
                logger.error(f'MLFlowException msg: {getattr(e, "message")}')
            