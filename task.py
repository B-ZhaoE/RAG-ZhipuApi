from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI
import faiss
import numpy as np

# 读取1.md文件中的内容
with open("1.md", "r", encoding="utf-8") as file:
    article = file.read()

# 初始化智谱API
client = ZhipuAI(api_key="de081fb1e13619e6d979ae271042eac5.a8whmZ0FG24RRPDF")

# 使用TextSplitter分割文章
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(article)

# 获取每个chunk的embedding
embeddings = []
for chunk in chunks:
    response = client.embeddings.create(
        model="embedding-2", 
        input=chunk
    )
    
    embedding = response.data[0].embedding
    embeddings.append(embedding)

# 将embeddings转换为NumPy数组
embedding_matrix = np.array(embeddings).astype('float32')

# 使用FAISS构建VectorStore（索引）
dimension = embedding_matrix.shape[1]  # 嵌入维度
index = faiss.IndexFlatL2(dimension)  # 使用L2距离度量
index.add(embedding_matrix)  # 添加embedding

# 保存索引到文件
faiss.write_index(index, 'faiss/faiss_index.faiss')

# 读取索引
index = faiss.read_index('faiss/faiss_index.faiss')

# 查询示例：搜索与某个query最相似的chunk
query = "解决冲突的过程通常包括什么步骤？"
response = client.embeddings.create(
    model="embedding-2",
    input=query
)
query_embedding = np.array(response.data[0].embedding).astype('float32')

# 使用FAISS进行相似度搜索
D, I = index.search(np.expand_dims(query_embedding, axis=0), k=3)  # 获取最相似的3个结果
print(I)  # 输出索引位置

similar_chunks = [chunks[i] for i in I[0]]
print(similar_chunks)


messages = [
    {"role": "user", "content": "使用以上下文来回答用户的问题。总是使用中文回答。"},
    {"role": "user", "content": "问题: "+query},
    {"role": "user", "content": "可参考的上下文:\n".join(similar_chunks)}  # 将相关的chunks作为上下文提供
]

response = client.chat.completions.create(
    model="glm-4-plus",
    messages=messages,
    max_tokens=200
)

# 输出生成的回答
print("\n生成的回答:", response.choices[0].message.content)