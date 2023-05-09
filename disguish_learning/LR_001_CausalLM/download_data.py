

cache_dir = '/root/autodl-tmp/data/'

from transformers import  AutoModel,AutoTokenizer
from datasets import  load_dataset

data = load_dataset("cahya/instructions-zh",cache_dir=cache_dir)
from transformers import AutoModelForSeq2SeqLM
m = AutoModelForSeq2SeqLM.from_pretrained('THUDM/glm-2b',cache_dir=cache_dir,
                                              trust_remote_code=True)