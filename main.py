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

# taining_data
# pt is pytorch.
# this will make sure all samples in our batch have the same length
# pytoch tensor returned directly. explianition to come
x_train = ["I love you", "i hate you"]
batch = tokenizer(x_train, padding=True, truncation=True, max_length=True, return_tensors='pt')

# disables the gradient tracking
with torch.no_grad():
    # ** means unpack
    outputs = model(**batch)