#-----------------------------------------------------------------------------------------
# Before running, authorize with huggingface token (with access to AI-Sweden-Models)
# https://huggingface.co/docs/huggingface_hub/quick-start#login
# Create token at https://huggingface.co/settings/tokens
# CLI: huggingface-cli login

# AI-Sweden-Models/gpt-sw3-126m is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
# If this is a private repository, make sure to pass a token having permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass `use_auth_token=True`.
#-----------------------------------------------------------------------------------------

from flask import Flask, make_response, request

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize Variables
# model_name = "AI-Sweden-Models/gpt-sw3-126m"
model_name = "AI-Sweden-Models/gpt-sw3-6.7b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)

generator = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)

def generateText(input):
    # generated = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=1)[0]["generated_text"]
    input_ids = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=1,
    )[0]

    generated_text = tokenizer.decode(generated_token_ids) 
    return generated_text

# test = generateText("Träd är fina för att")
app = Flask(__name__)

@app.route("/")
def hello():
    rPrompt = request.args.get("prompt")
    if not(rPrompt is None) and len(rPrompt):
        response = make_response(generateText(rPrompt), 200)
    else:
        response = make_response("prompt parameter empty", 200)
    
    response.mimetype = "text/plain"
    return response
