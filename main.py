from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import pandas as pd

csv = pd.read_csv('tezos_comments.csv')
df = pd.DataFrame(csv)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)


results = classifier(["I love you", "i hate you"])
# for result in results:
#     print(result)

tokens = tokenizer.tokenize("In this video I show you everything to get started with")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token2_ids = tokenizer("In this video I show you everything to get started with")

print(tokens)
print(token_ids)
print(token2_ids)



