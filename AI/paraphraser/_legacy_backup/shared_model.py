from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


model_name = "google/flan-t5-large"
print(f"Caricamento modello e tokenizer: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


if tokenizer.pad_token is None or tokenizer.pad_token != '<pad>':
    print(
        f"tokenizer.pad_token was: {tokenizer.pad_token}. Adding '<pad>' token.")
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model.resize_token_embeddings(len(tokenizer))
else:
    print(
        f"tokenizer.pad_token is already correctly set to: '{tokenizer.pad_token}' with ID: {tokenizer.pad_token_id}")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modello spostato su: {device}")
