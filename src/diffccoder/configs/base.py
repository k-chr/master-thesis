from dataclasses import dataclass
from enum import Enum
from pathlib import Path, WindowsPath, PosixPath

import yaml

import diffccoder.configs.enums as enums

YAML_INITALIZED = False


class ConfigDumper(yaml.Dumper):
    def increase_indent(self, **_):
        return super().increase_indent(flow=True, indentless=False)


def enum_loader(e: type[Enum]):
    def __lambda(loader: yaml.Loader, node: yaml.Node):
        return e[(str.upper(loader.construct_mapping(node)['value']))]
    return __lambda

def init_enums():
    for _e_type in enums.__enums__:
        yaml.add_representer(_e_type, map_enum, Dumper=ConfigDumper) 
        yaml.add_constructor('!' + _e_type.__name__,
                             enum_loader(_e_type),
                             Loader=yaml.Loader)

def init_paths():
    yaml.add_representer(PosixPath, 
                         map_path,
                         Dumper=ConfigDumper)
    yaml.add_representer(WindowsPath, 
                         map_path,
                         Dumper=ConfigDumper)
    yaml.add_constructor('!Path',
                         lambda loader, node:
                            Path(loader.construct_mapping(node)['data']),
                         Loader= yaml.Loader)

def map_path(dumper: ConfigDumper, data: Path):
    return dumper.represent_mapping('!Path', {'data':data.__str__()})

def map_enum(dumper: ConfigDumper, data: Enum):
    return dumper.represent_mapping('!' + data.__class__.__name__,
                                    {'value' : data.name.lower()})
    
def yaml_module_init():
    
    if YAML_INITALIZED: return 
    yaml.Dumper.ignore_aliases = lambda *_ : True 
    yaml.add_representer(tuple, 
                         lambda dumper, data: 
                             dumper.represent_sequence('!tuple', (data), True),
                         Dumper=ConfigDumper)
    yaml.add_representer(list, 
                         lambda dumper, data: 
                             dumper.represent_sequence('!list', (data), True),
                         Dumper=ConfigDumper)
    yaml.add_constructor('!tuple',
                         lambda loader, node:
                            (loader.construct_sequence(node)),
                         Loader= yaml.Loader)
    yaml.add_constructor('!list',
                         lambda loader, node:
                            list(loader.construct_sequence(node)),
                         Loader= yaml.Loader)
    init_paths()
    init_enums()

    
class _Base(yaml.YAMLObject):
    __registered_types: set['_Base'] = set()
    yaml_tag: str = '!Base'
    
    def register(self) -> None:
        return self.__class__._register()
    
    @classmethod
    def _register(cls) -> None:
        global YAML_INITALIZED
        if not YAML_INITALIZED:
            yaml_module_init()
            YAML_INITALIZED = True
            
        cls.yaml_tag = '!' + cls.__name__
        _Base.__registered_types.add(cls)
        
        yaml.add_constructor(cls.yaml_tag, 
                             lambda loader, node:
                                cls(**loader.construct_mapping(node)),
                             Loader=yaml.Loader)

        yaml.add_representer(cls,
                             lambda dumper, data: 
                                dumper.represent_mapping(cls.yaml_tag, data.__dict__),
                             Dumper=ConfigDumper)
        
    @classmethod
    def registered_configs(cls): return _Base.__registered_types
    

@dataclass
class BaseConfig(_Base):
    def __init_subclass__(cls) -> None:
        cls._register()
    def __post_init__(self):
        self.register()
        
def dump_config(config: BaseConfig, out_dir: Path, indent: int = 4):
    f_name = config.yaml_tag[1:].lower() + '.yaml'
    with open(out_dir / f_name, 'w') as out:
        yaml.dump(config, indent=indent, stream=out, Dumper=ConfigDumper)
        
def load_config(path_to_yaml: Path) -> BaseConfig:
    with open(path_to_yaml) as f:
        obj =  yaml.load(f, yaml.Loader)
    return obj
