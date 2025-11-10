from transformers import AutoModelForTokenClassification, AutoTokenizer

checkpoint = "checkpoints/drbert"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint)
