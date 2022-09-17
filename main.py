from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import pandas as pd

# csv = pd.read_csv('tezos_comments.csv')
# df = pd.DataFrame(csv)
#
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
#
#
# results = classifier(["I love you", "i hate you"])
# for result in results:

#     print(result)

# tokens = tokenizer.tokenize("In this video I show you everything to get started with")
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# token2_ids = tokenizer("In this video I show you everything to get started with")
#
# # print(tokens)
# # print(token_ids)
# # print(token2_ids)
#
# # taining_data
# # pt is pytorch.
# # this will make sure all samples in our batch have the same length
# # pytoch tensor returned directly. explianition to come
# x_train = ["I love you", "i hate you"]
# batch = tokenizer(x_train, padding=True, truncation=True, max_length=True, return_tensors='pt')


# def fine_tune():
#     """if you do it this way, its technically the same, but more fine tuned"""
#     # disables the gradient tracking
#     with torch.no_grad():
#         # ** means unpack
#         outputs = model(**batch, labels=torch.tensor([1, 0]))
#         # ?this will be raw values
#         print(outputs)
#         predictions = F.softmax(outputs.logits, dim=1)
#         print(predictions)
#         # index with the highest probability
#         labels = torch.argmax(predictions, dim=1)
#         print(labels)
#         labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
#         print(labels)

# save_dir = "saved"
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)
model_name =  f"cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

import pandas as pd

csv = pd.read_csv('tezos_comments.csv')
df = pd.DataFrame(csv)

text = df.iloc[0, 1]

batch = tokenizer(text, padding=True, truncation=True, max_length=True, return_tensors='pt')
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs.logits, dim = 1)
    print(label_ids)
    labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    print(labels)
