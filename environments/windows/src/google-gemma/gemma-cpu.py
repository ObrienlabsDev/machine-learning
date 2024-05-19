import os
# default dual GPU - either PCIe bus or NVidia bus - slowdowns
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# specific GPU - model must fit entierely in memory RTX-3500 ada = 12G, A4000=16G, A4500=20, A6000=48, 4000 ada = 20, 5000 ada = 32, 6000 ada = 48
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

access_token='hf_...qH'

model = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
# GPU
#model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", token=access_token)
# CPU
model = AutoModelForCausalLM.from_pretrained(model,token=access_token)

input_text = "how is gold made in collapsing neutron stars - specifically what is the ratio created during the beta and r process."
time_start = datetime.now().strftime("%H:%M:%S")
print("genarate start: ", datetime.now().strftime("%H:%M:%S"))

# GPU
#input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
# CPU
input_ids = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**input_ids, 
                         max_new_tokens=10000)
print(tokenizer.decode(outputs[0]))

print("end", datetime.now().strftime("%H:%M:%S"))
time_end = datetime.now().strftime("%H:%M:%S")
