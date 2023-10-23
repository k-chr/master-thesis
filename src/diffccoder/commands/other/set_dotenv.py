import os
from pathlib import Path

from cleo.commands.command import Command
from cleo.helpers import argument


class SetDotEnvCommand(Command):
    name = 'set-dotenv'
    description = 'set_dotenv.py - Exports provided enviroment variables to .env file.'
    arguments = [argument('env-var',
                          multiple=True,
                          description='Variables to store.')]

    def handle(self) -> int:
        variables: list[str] = self.argument('env-var')
        path: Path = Path('~/.env')
        
        with open(path, 'w', encoding='utf8') as f:
            for var in variables:
                f.write(f'{var}={os.environ.get(var, "")}\n')
