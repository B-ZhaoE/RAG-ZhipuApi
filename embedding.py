from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 加载1.md文件内容
with open("1.md", "r", encoding="utf-8") as file:
    document = file.read()

# 分词并将文本转换为模型输入
inputs = tokenizer(document, return_tensors="pt", truncation=True, padding=True)

# 获取模型输出，不需要计算梯度
with torch.no_grad():
    outputs = model(**inputs)

# 获取[CLS]位置的embedding或平均pooling后的嵌入
embedding = outputs.last_hidden_state.mean(dim=1)  # 平均pooling

# 将embedding转换为numpy数组并保存到本地
embedding_numpy = embedding.cpu().numpy()
np.save("embedding\embedding.npy", embedding_numpy)

