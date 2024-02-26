from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

#access_token='hf_cfTP...XCQqH'


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=access_token)
# GPU
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", token=access_token)
# CPU
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",token=access_token)

input_text = "how is gold made in collapsing neutron stars - specifically what is the ratio created during the beta and r process."
time_start = datetime.now().strftime("%H:%M:%S")
print("genarate start: ", datetime.now().strftime("%H:%M:%S"))

# GPU
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
# CPU
#input_ids = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**input_ids, 
                         max_new_tokens=10000)
print(tokenizer.decode(outputs[0]))

print("end", datetime.now().strftime("%H:%M:%S"))
time_end = datetime.now().strftime("%H:%M:%S")