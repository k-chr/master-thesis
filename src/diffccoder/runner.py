from cleo.application import Application

from diffccoder.commands.data.chunkify_large_txt import ChunkifyLargeTextCommand
from diffccoder.commands.data.extract_parquet import ExtractParquetCommand
from diffccoder.commands.data.process_small_xml import ProcessSmallXMLCommand
from diffccoder.commands.data.tokenize_dante import TokenizeDanteCommand
from diffccoder.commands.data.tokenize_files import TokenizeFilesCommand
from diffccoder.commands.data.train_tokenizer import TrainTokenizerCommand
from diffccoder.commands.model.run_fine_tuning import DiffFineTuningCommand
from diffccoder.commands.model.run_inference import InferCommand
from diffccoder.commands.model.run_pretraining import PreTrainingCommand
from diffccoder.commands.model.run_diffusion_training import DiffTrainingCommand
from diffccoder.commands.other.generate_template_yamls import GenerateTemplateYamlsCommand
from diffccoder.commands.other.create_experiment import CreateExperimentCommand
from diffccoder.commands.other.create_or_clone_run import CreateOrCloneRunCommand
from diffccoder.commands.other.set_dotenv import SetDotEnvCommand
from diffccoder.commands.other.mlflow_updater import MlFlowUpdaterCommand

app = Application('runner', '0.0.1')
app.add(ExtractParquetCommand())
app.add(ProcessSmallXMLCommand())
app.add(ChunkifyLargeTextCommand())
app.add(TokenizeFilesCommand())
app.add(TrainTokenizerCommand())
app.add(PreTrainingCommand())
app.add(DiffTrainingCommand())
app.add(DiffFineTuningCommand())
app.add(InferCommand())
app.add(GenerateTemplateYamlsCommand())
app.add(CreateExperimentCommand())
app.add(CreateOrCloneRunCommand())
app.add(SetDotEnvCommand())
app.add(MlFlowUpdaterCommand())
app.add(TokenizeDanteCommand())