from sentence_transformers import SentenceTransformer

print("ruri-largeをダウンロード中...")
model = SentenceTransformer("cl-nagoya/ruri-large")

# 動作確認
sentences = ["RAGとは検索拡張生成の技術です。", "半導体製造工程の検査方法について"]
embeddings = model.encode(sentences)

print(f"成功：embedding shape = {embeddings.shape}")
