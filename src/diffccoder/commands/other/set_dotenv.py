import os
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument
from loguru import logger


class SetDotEnvCommand(Command):
    name = 'set-dotenv'
    description = 'set_dotenv.py - Exports provided enviroment variables to .env file.'
    arguments = [argument('env-var',
                          multiple=True,
                          description='Variables to store.')]

    def handle(self) -> int:
        variables: list[str] = self.argument('env-var')
        logger.info(f'Saving selected environment variables to file, variables: {variables}')
        path: Path = Path.home() / '.env'
        
        with open(path, 'w', encoding='utf8') as f:
            for var in variables:
                f.write(f'{var}={os.environ[var]}\n')
