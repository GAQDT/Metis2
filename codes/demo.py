from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_serving.model.wrapper import get_model

# Load the tokenizer from the local path where config.json is stored
tokenizer = AutoTokenizer.from_pretrained("/root/models/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6", local_files_only=True)
tokenizer.add_bos_token = False

# Load the model from the local path
model = AutoModelForCausalLM.from_pretrained("/root/models/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6", local_files_only=True)

# Generate
prompt = "Who are you"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)
