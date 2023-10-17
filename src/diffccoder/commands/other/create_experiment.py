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

        config_dir = Path('.').home() / 'share' / 'exp' / exp_name / 'template_configs'
        
        logger.info('Calling command "generate-template-configs" for new experiment.')
        self.call('generate-template-configs:handle', config_dir.__str__())
        
        exp_config: ExperimentConfig = load_config(config_dir / 'experimentconfig.yaml')
        
        exp_config.exp_root = config_dir.parent
        exp_config.experiment_name = exp_name
        
        dump_config(exp_config, config_dir)

        if init_run:
            logger.info(f'Creating a new run for current experiment: {exp_name}.')
            exp_config.work_dir = exp_config.exp_root / '000000'
            exp_config.work_dir.mkdir(parents=True, exist_ok=True)
            root = exp_config.exp_root
            exp_config.checkpoint_dir = root / 'checkpoints'
            exp_config.data_dir = root.home() / 'share' / 'data' / 'tokenized'
            exp_config.out_dir = root / 'out'
            _cfg_dir = exp_config.work_dir / 'configs'
            
            shutil.copytree(config_dir, _cfg_dir, dirs_exist_ok=True)
            logger.info('Copied config templates to brand-new run directory tree.')
            dump_config(exp_config, _cfg_dir)
        
        if use_mlflow:
            return self._setup_mlflow_exp(exp_name, exp_config, _cfg_dir)

    def _setup_mlflow_exp(self, exp_name: str, exp_config: ExperimentConfig, config_dir: Path):
        logger.debug(f'Environmental variables: {os.environ}')
        
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        logger.info(f'Checking if experiment already exists in: {os.environ["MLFLOW_TRACKING_URI"]} server')
        
        exp = mlflow.get_experiment_by_name(exp_name)

        if not exp:
            exp_config.mlflow_enabled = True
            exp_config.mlflow_server = os.environ['MLFLOW_TRACKING_URI']
            logger.info(f'Creating new experiment for: {os.environ["MLFLOW_TRACKING_URI"]} server')
            exp = mlflow.create_experiment(exp_name, artifact_location=exp_config.exp_root)
                
            with mlflow.start_run(experiment_id=exp, run_name='000000') as run:
                exp_config.mlflow_run_id = run.info.run_id
                exp_config.mlflow_run_name = f'{run.info.run_name}'
                        
            dump_config(exp_config, config_dir)
            
            logger.success('Successfully updated run config')
                   
        else:
            logger.error(f'Experiment: {exp_name} exists!!!')
            return -1

    def handle(self) -> int:
        
        exp_name = self.argument('experiment-name')
        return self._create_exp(exp_name, self.option('mlflow'), self.option('init-run'))
    