

cache_dir = '/root/autodl-tmp/data/'

from transformers import  AutoModel,AutoTokenizer
from datasets import  load_dataset

data = load_dataset("cahya/instructions-zh",cache_dir=cache_dir)