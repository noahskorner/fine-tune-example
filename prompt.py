from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Load the fine-tuned model and tokenizer
# model_path = "output/fine-tuned-model"
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Function to get the model's response
def get_response(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the output
    output = model.generate(
        **inputs,
        max_new_tokens=100,     # Limits the length of the generated response
        temperature=0.7,        # Adjusts creativity level
        top_p=0.9,              # Nucleus sampling for more diverse responses
        do_sample=True,         # Enables sampling for varied outputs
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
    )

    # Decode and clean up the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

dialog = ""
# Interactive conversation loop
while True:
    user_input = input("Character B: ")
    dialog += f"Character B: {user_input}\n"
    prompt = (
        "You are Character A, a friend of Character B. What does Character A say next?"
        f"\n{dialog}"
    )

    # Get the model's response
    response = get_response(prompt)
    print(response)
    dialog += f"\nresponse"
