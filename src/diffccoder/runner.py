from cleo.application import Application

from diffccoder.commands.data.chunkify_large_txt import ChunkifyLargeTextCommand
from diffccoder.commands.data.extract_parquet import ExtractParquetCommand
from diffccoder.commands.data.process_small_xml import ProcessSmallXMLCommand
from diffccoder.commands.data.tokenize_files import TokenizeFilesCommand
from diffccoder.commands.data.train_tokenizer import TrainTokenizerCommand
from diffccoder.commands.model.run_pretraining import PreTrainingCommand
from diffccoder.commands.other.generate_template_yamls import GenerateTemplateYamlsCommand

app = Application('runner', '0.0.1')
app.add(ExtractParquetCommand())
app.add(ProcessSmallXMLCommand())
app.add(ChunkifyLargeTextCommand())
app.add(TokenizeFilesCommand())
app.add(TrainTokenizerCommand())
app.add(PreTrainingCommand())
app.add(GenerateTemplateYamlsCommand())
