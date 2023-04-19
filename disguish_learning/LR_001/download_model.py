

cache_dir = '/root/autodl-tmp/model/'

from transformers import  AutoModel,AutoTokenizer
model = AutoModel.from_pretrained('THUDM/glm-10b',cache_dir=cache_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-2b',cache_dir=cache_dir,trust_remote_code=True)