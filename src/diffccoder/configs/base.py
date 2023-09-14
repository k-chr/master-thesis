from dataclasses import dataclass
from pathlib import Path, WindowsPath, PosixPath

import yaml

YAML_INITALIZED = False


class ConfigDumper(yaml.Dumper):
    def increase_indent(self, **_):
        return super().increase_indent(flow=True, indentless=False)
    

def yaml_module_init():
    global YAML_INITALIZED
    if YAML_INITALIZED: return 
    yaml.Dumper.ignore_aliases = lambda *_ : True 
    yaml.add_representer(PosixPath, 
                         lambda dumper, data:
                            dumper.represent_mapping("!Path", {'data':data.__str__()}),
                         Dumper=ConfigDumper)
    yaml.add_representer(WindowsPath, 
                         lambda dumper, data:
                            dumper.represent_mapping("!Path", {'data':data.__str__()}),
                         Dumper=ConfigDumper)
    yaml.add_constructor('!Path',
                         lambda loader, node:
                            Path(loader.construct_mapping(node)['data']),
                         Loader= yaml.Loader)
    
    
class _Base(yaml.YAMLObject):
    __registered_types: set['_Base'] = set()
    yaml_tag: str = '!Base'
    
    def register(self) -> None:
        return self.__class__._register()
    
    @classmethod
    def _register(cls) -> None:
        global YAML_INITALIZED
        if not YAML_INITALIZED:
            YAML_INITALIZED = True
            yaml_module_init()
            
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
        yaml.dump(config, indent=indent, stream = out, Dumper=ConfigDumper)
        
def load_config(path_to_yaml: Path) -> BaseConfig:
    with open(path_to_yaml) as f:
        obj =  yaml.load(f, yaml.Loader)
    return obj
