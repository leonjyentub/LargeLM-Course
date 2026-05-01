import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # 1. 載入預訓練的 GloVe 權重 (使用較輕量的 100 維版本)
    print("正在下載/載入 GloVe 預訓練模型 (glove-wiki-gigaword-100)...")
    model = api.load("glove-wiki-gigaword-100")

    # 2. 計算 "Programmer" 與性別詞彙的距離 (使用餘弦相似度)
    # 注意：cosine_similarity 越高代表距離越近
    target_word = "programmer"
    sim_man = model.similarity(target_word, "man")
    sim_woman = model.similarity(target_word, "woman")

    print(f"\n--- 距離計算結果 ---")
    print(f"'{target_word}' 與 'man' 的相似度: {sim_man:.4f}")
    print(f"'{target_word}' 與 'woman' 的相似度: {sim_woman:.4f}")
    print(f"偏向度: {'男性' if sim_man > sim_woman else '女性'} (差值: {abs(sim_man - sim_woman):.4f})")

    # 3. 找出與 "Gender" 維度最強關聯的職業
    # 定義性別維度向量: g = v(woman) - v(man)
    gender_vector = model['woman'] - model['man']

    # 定義一組職業清單進行測試
    professions = [
        "nurse", "doctor", "programmer", "teacher", "engineer", 
        "secretary", "soldier", "homemaker", "scientist", "dancer",
        "pilot", "librarian", "lawyer", "assistant", "manager"
    ]

    # 計算各職業向量與性別維度的投影 (相似度)
    results = []
    for p in professions:
        if p in model:
            # 計算該職業向量與性別維度向量的餘弦相似度
            # 正值越高代表越偏向女性，負值越低代表越偏向男性
            vec_p = model[p].reshape(1, -1)
            vec_g = gender_vector.reshape(1, -1)
            score = cosine_similarity(vec_p, vec_g)[0][0]
            results.append((p, score))

    #