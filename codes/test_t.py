import torch
import torch.nn as nn
from transformers import OPTForCausalLM, OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import time
import json
import os
import tensor_parallel as tp



tp = 2  # 张量并行度
bs = 2  # 批量大小

# 使用Hugging Face的OPT-3.1B模型
config = OPTConfig.from_pretrained("facebook/opt-125m")
model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
print("1")
print(model)

import transformers
import tensor_parallel as tp
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-125m")  # use opt-125m for testing
print("2")
print(model)
