from transformers import AutoTokenizer
import transformers
import torch

model = "../model/llama-2-13b-hf"


tokenizer = AutoTokenizer.from_pretrained(model)
generator = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(torch.cuda.current_device())
input_text= 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
output_text = generator(
    input_text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    max_length=100,
)
print(output_text[0]['generated_text'])


