from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM

model_weight = "../model/llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_weight)
model = LlamaForCausalLM.from_pretrained(model_weight)

input_text = "Hello, I am a"
inputs = tokenizer(input_text, return_tensors='pt')
print(inputs)
outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
print('Generated text:')
for i, output in enumerate(outputs):
    print(f'{i}: {tokenizer.decode(output)}')