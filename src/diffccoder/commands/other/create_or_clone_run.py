import os
from pathlib import Path
import shutil

from cleo.commands.command import Command
from cleo.helpers import argument
from loguru import logger
import mlflow

from diffccoder.configs.base import dump_config, load_config
from diffccoder.configs.experiment_config import ExperimentConfig


class CreateOrCloneRunCommand(Command):
    name = 'create-or-clone-run'
    description = 'create-or-clone-run.py Creates or clones run.'
    arguments = [argument('experiment-name',
                          description='Name of an experiment'),
                 argument('run-name',
                          description='Name of an run to start',
                          optional=True),
                 argument('run-to-clone',
                          description='Name of a run to clone',
                          optional=True)]
    
    def _clone_run(self, exp_name: str, src_run: str, dst_run: str):
        logger.info(f'Cloning configuration from {exp_name}/{src_run} to {exp_name}/{dst_run}')

        root_config_dir = Path.home() / 'share' / 'exp' / exp_name / 'template_configs'
        
        logger.info('Load experiment template config')
        exp_config: ExperimentConfig = load_config(root_config_dir / 'experimentconfig.yaml')
        src_config_dir = exp_config.exp_root / 'runs' / src_run / 'configs'
        dst_config_dir = exp_config.exp_root / 'runs' / dst_run / 'configs'
        
        dst_config_dir.mkdir(parents=True, exist_ok=True)
        
        src_config: ExperimentConfig = load_config(src_config_dir / 'experimentconfig.yaml')
        
        shutil.copytree(src_config_dir, dst_config_dir, dirs_exist_ok=True)
        
        src_config.work_dir = dst_config_dir.parent
        src_config.checkpoint_dir = src_config.work_dir / 'checkpoints'
        src_config.out_dir = src_config.work_dir / 'out'
        
        use_mlflow = src_config.mlflow_enabled
        logger.info('Copied config templates to destination run directory tree.')
        
        dump_config(src_config, dst_config_dir)
        
        if use_mlflow:
            return self._setup_mlflow_run(exp_name, dst_run, src_config, dst_config_dir)
        
    def _create_run(self, exp_name: str, run_name: str):
        logger.info(f'Creating a run with name: {run_name}')

        root_config_dir = Path.home() / 'share' / 'exp' / exp_name / 'template_configs'
        
        logger.info('Load experiment template config')
        exp_config: ExperimentConfig = load_config(root_config_dir / 'experimentconfig.yaml')
        
        logger.info(f'Creating a new run {run_name} for current experiment: {exp_name}.')
        exp_config.work_dir = exp_config.exp_root / 'runs' / run_name
        exp_config.work_dir.mkdir(parents=True, exist_ok=True)
        exp_config.checkpoint_dir = exp_config.work_dir / 'checkpoints'
        exp_config.data_dir = Path.home() / 'share' / 'data'
        exp_config.out_dir = exp_config.work_dir / 'out'
        config_dir = exp_config.work_dir / 'configs'
        
        shutil.copytree(root_config_dir, config_dir, dirs_exist_ok=True)
        logger.info('Copied config templates to brand-new run directory tree.')
        dump_config(exp_config, config_dir)
        
        use_mlflow = exp_config.mlflow_enabled
        
        if use_mlflow:
            return self._setup_mlflow_run(exp_name, run_name, exp_config, config_dir)

    def _setup_mlflow_run(self, exp_name: str, run_name: str, exp_config: ExperimentConfig, config_dir: Path):
        logger.debug(f'Environmental variables: {os.environ}')
        
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        
        logger.info(f'Checking if experiment already exists in: {os.environ["MLFLOW_TRACKING_URI"]} server')
        
        exp = mlflow.get_experiment_by_name(exp_name)

        if exp:                
            with mlflow.start_run(experiment_id=exp, run_name=run_name) as run:
                exp_config.mlflow_run_id = run.info.run_id
                exp_config.mlflow_run_name = f'{run.info.run_name}'
                        
            dump_config(exp_config, config_dir)
            logger.success('Successfully updated run config')
                   
        else:
            logger.error(f'Experiment: {exp_name} should exist!!!')
            return -1

    def _get_new_run_name(exp_name: str) -> str:
        runs_root = Path.home() / 'share' / 'exp' / exp_name / 'runs'
        number = 0
        if runs_root.is_dir():
        
            last_run = list(runs_root.iterdir())[-1]

            number = int(last_run.stem) + 1

        return str(number).zfill(6)  

    def handle(self) -> int:
        
        exp_name = self.argument('experiment-name')
        run_name = self.argument('run-name') or self._get_new_run_name(exp_name)
        run_to_clone = self.argument('run-to-clone')
        
        if run_to_clone:
            assert run_name != run_to_clone
            self._clone_run(exp_name, run_to_clone, run_name)
        
        return self._create_run(exp_name, run_name)
    