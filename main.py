from transformers import pipeline

import torch
import torch.nn.functional as F

classifier = pipeline("sentiment-analysis")

res = classifier(["Congratulations! You have been selected.",
                  "I hope you dont hate it."])
for result in res:
  print(result)
