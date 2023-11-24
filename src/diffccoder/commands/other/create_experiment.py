import os
from pathlib import Path
import shutil

from cleo.commands.command import Command
from cleo.helpers import argument, option
from loguru import logger
import mlflow

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.experiment_config import ExperimentConfig

template_configs = list(filter(lambda parent:  parent.stem == 'master-thesis', Path(__file__).resolve().parents))[0] / 'templates' / 'configs'
print(template_configs)

class CreateExperimentCommand(Command):
    name = 'create-experiment'
    description = 'create_experiment.py Creates experiment.'
    arguments = [argument('experiment-name',
                          description='Name of an experiment to create')]
    options = [option('mlflow', 'm',
                      description='Log an experiment to mlflow.'),
               option('copy-templates', 'c',
                      description='Copy ready-to-go templates instead of config generation'),
               option('init-run', 'i',
                      description='Init mlflow experiment with empty run.')]

    def _create_exp(self, exp_name: str, use_mlflow: bool =True, init_run: bool =False, copy_templates: bool =False):
        logger.info(f'Creating an experiment with name: {exp_name}')

        config_dir = Path.home() / 'share' / 'exp' / exp_name / 'template_configs'
        
        if not copy_templates:
            logger.info('Calling command "generate-template-yamls" for new experiment.')
            self.call('generate-template-yamls', f'PLACEHOLDER {config_dir.__str__()}')        
            
        else:
            shutil.copytree(src=template_configs, dst=config_dir, dirs_exist_ok=True)
            
        exp_config: ExperimentConfig = load_config(config_dir / 'experimentconfig.yaml')
        
        exp_config.exp_root = config_dir.parent
        exp_config.mlflow_enabled = use_mlflow
        exp_config.experiment_name = exp_name
        
        dump_config(exp_config, config_dir)
        
        if use_mlflow:
            self._setup_mlflow_exp(exp_name, exp_config, config_dir)

        if init_run:
            self.call('create-or-clone-run', f'PLACEHOLDER {exp_name}')

    def _setup_mlflow_exp(self, exp_name: str, exp_config: ExperimentConfig, config_dir: Path):
        logger.debug(f'Environmental variables: {os.environ}')
        
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        logger.info(f'Checking if experiment already exists in: {os.environ["MLFLOW_TRACKING_URI"]} server')
        
        exp = mlflow.get_experiment_by_name(exp_name)

        if not exp:
            exp_config.mlflow_enabled = True
            exp_config.mlflow_server = os.environ['MLFLOW_TRACKING_URI']
            logger.info(f'Creating new experiment for: {os.environ["MLFLOW_TRACKING_URI"]} server')
            exp_id = mlflow.create_experiment(exp_name, artifact_location=exp_config.exp_root.__str__())
           
            dump_config(exp_config, config_dir)
            logger.success(f'Successfully created new mlflow experiment: {exp_name} with id: {exp_id}')  
        else:
            logger.error(f'Experiment: {exp_name} exists!!!')
            return -1

    def handle(self) -> int:
        
        exp_name = self.argument('experiment-name')
        return self._create_exp(exp_name, self.option('mlflow'), self.option('init-run'), self.option('copy-tempaltes'))
    