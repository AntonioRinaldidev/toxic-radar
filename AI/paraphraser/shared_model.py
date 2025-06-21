from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


model_name = "RinaldiDev/flan-polite-finetuned"
print(f"Caricamento modello e tokenizer: {model_name}")
polish_tokenizer = T5Tokenizer.from_pretrained(model_name)
polish_model = T5ForConditionalGeneration.from_pretrained(model_name)


if polish_tokenizer.pad_token is None or polish_tokenizer.pad_token != '<pad>':
    print(
        f"tokenizer.pad_token was: {polish_tokenizer.pad_token}. Adding '<pad>' token.")
    polish_tokenizer.add_special_tokens({'pad_token': '<pad>'})

    polish_model.resize_token_embeddings(len(polish_tokenizer))
else:
    print(
        f"tokenizer.pad_token is already correctly set to: '{polish_tokenizer.pad_token}' with ID: {polish_tokenizer.pad_token_id}")


device = "cuda" if torch.cuda.is_available() else "cpu"
polish_model.to(device)
print(f"Modello spostato su: {device}")
