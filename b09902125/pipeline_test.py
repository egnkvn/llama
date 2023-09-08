from transformers import AutoTokenizer
import transformers
import torch

model = "../model/llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

input_text = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
input_text_cn = '請生成"讓子彈飛"中，關於胡萬與老六的對話\n胡萬：你給了一碗的錢吃了兩碗的粉\n老六：'
input_text_jp = '私は陳です　お名前は?\n'

sequences = pipeline(
    input_text_jp,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")