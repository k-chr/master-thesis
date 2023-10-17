import os
from pathlib import Path
import shutil

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
import mlflow

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.experiment_config import ExperimentConfig


class CreateExperimentCommand(Command):
    name = 'create-experiment'
    description = 'create_experiment.py Creates experiment.'
    arguments = [argument('experiment-name',
                          description='Name of an experiment to create')]
    options = [option('mlflow', 'm',
                      description='Log an experiment to mlflow.'),
               option('init-run', 'i',
                      description='Init mlflow experiment with empty run.')]

    def _create_exp(self, exp_name: str, use_mlflow: bool =True, init_run: bool =False):
        logger.info(f'Creating an experiment with name: {exp_name}')

        config_dir = Path.home() / 'share' / 'exp' / exp_name / 'template_configs'
        
        logger.info('Calling command "generate-template-configs" for new experiment.')
        self.call('generate-template-configs:handle', config_dir.__str__())
        
        exp_config: ExperimentConfig = load_config(config_dir / 'experimentconfig.yaml')
        
        exp_config.exp_root = config_dir.parent
        exp_config.experiment_name = exp_name
        
        dump_config(exp_config, config_dir)
        
        if use_mlflow:
            return self._setup_mlflow_exp(exp_name, exp_config, config_dir)

        if init_run:
            self.call('create-or-clone-run:handle', f'{exp_name} 000000')

    def _setup_mlflow_exp(self, exp_name: str, exp_config: ExperimentConfig, config_dir: Path):
        logger.debug(f'Environmental variables: {os.environ}')
        
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        logger.info(f'Checking if experiment already exists in: {os.environ["MLFLOW_TRACKING_URI"]} server')
        
        exp = mlflow.get_experiment_by_name(exp_name)

        if not exp:
            exp_config.mlflow_enabled = True
            exp_config.mlflow_server = os.environ['MLFLOW_TRACKING_URI']
            logger.info(f'Creating new experiment for: {os.environ["MLFLOW_TRACKING_URI"]} server')
            exp_id = mlflow.create_experiment(exp_name, artifact_location=exp_config.exp_root)
           
            dump_config(exp_config, config_dir)
            logger.success(f'Successfully created new mlflow experiment: {exp_name} with id: {exp_id}')  
        else:
            logger.error(f'Experiment: {exp_name} exists!!!')
            return -1

    def handle(self) -> int:
        
        exp_name = self.argument('experiment-name')
        return self._create_exp(exp_name, self.option('mlflow'), self.option('init-run'))
    