#  ================ github Streamlit 部署 ===========

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List, Dict
from dotenv import load_dotenv
import faiss
import os
import json
from pathlib import Path
from openai import OpenAI
from rank_bm25 import BM25Okapi


class QwenRAGSystemOptimized:
    def __init__(self, api_key: str = None):
        # ========== API Key ==========
        """
        api_key: 用户可手动传入 Qwen API Key，如果不传则读取环境变量 DASHSCOPE_API_KEY
        """
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DASHSCOPE_API_KEY，请传入或设置环境变量"
            )

        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )

        # ========== 索引和元数据 ==========
        if not Path("vector_index.faiss").exists():
            raise FileNotFoundError("缺少 vector_index.faiss，请检查路径")
        if not Path("chunk_metadata.json").exists():
            raise FileNotFoundError("缺少 chunk_metadata.json，请检查路径")

        self.index = faiss.read_index("vector_index.faiss")
        with open("chunk_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # ========== 嵌入模型 ==========
        self.embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

        # ========== 交叉编码器 ==========
        self.cross_encoder = CrossEncoder("shibing624/text2vec-base-chinese")

        # ========== 初始化 BM25 ==========
        self._prepare_bm25_corpus()

    def _prepare_bm25_corpus(self):
        """预处理BM25检索语料"""
        if not self.metadata:
            raise ValueError("metadata 为空，无法初始化 BM25")

        self.all_texts = [item["text"] for item in self.metadata]
        # 使用 jieba 进行分词
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.all_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def hybrid_retrieve(self, query: str, k: int = 10, bm25_candidates: int = 100) -> List[Dict]:
        """混合检索：BM25初筛 + 向量检索 + 交叉编码器重排序"""
        if not hasattr(self, "bm25"):
            raise RuntimeError("BM25 尚未初始化，请检查 _prepare_bm25_corpus()")

        # 1. BM25 初筛
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_candidates]

        # 2. 向量检索
        query_embed = self.embed_model.encode([query])
        candidate_texts = [self.all_texts[i] for i in top_bm25_indices]
        candidate_embeddings = self.embed_model.encode(candidate_texts)
        vector_scores = cosine_similarity(query_embed, candidate_embeddings)[0]

        # 3. 融合 BM25 和向量得分
        bm25_normalized = (bm25_scores[top_bm25_indices] - np.min(bm25_scores[top_bm25_indices])) / (
            np.max(bm25_scores[top_bm25_indices]) - np.min(bm25_scores[top_bm25_indices]) + 1e-8
        )
        vector_normalized = (vector_scores - np.min(vector_scores)) / (
            np.max(vector_scores) - np.min(vector_scores) + 1e-8
        )

        hybrid_scores = 0.3 * bm25_normalized + 0.7 * vector_normalized
        top_hybrid_indices = top_bm25_indices[np.argsort(hybrid_scores)[::-1][:k]]

        # 4. 交叉编码器重排序
        rerank_pairs = [(query, self.all_texts[i]) for i in top_hybrid_indices]
        rerank_scores = self._batch_rerank(rerank_pairs)

        results = []
        for idx, score in zip(top_hybrid_indices, rerank_scores):
            chunk_data = {
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx]["metadata"],
                "score": float(score)
            }
            results.append(chunk_data)

        return results

    def _batch_rerank(self, rerank_pairs: List[tuple]) -> List[float]:
        """批量处理交叉编码器以加速重排序"""
        with ThreadPoolExecutor() as executor:
            rerank_scores = list(executor.map(self.cross_encoder.predict, rerank_pairs))
        return rerank_scores

    def generate_prompt(self, query: str, chunks: List[Dict]) -> str:
        """构建增强提示"""
        context = "\n\n".join([
            f"[来源：{c['metadata']['team_name']} {c['metadata']['year']}]\n{c['text']}"
            for c in chunks
        ])
        return f"""你是一名iGEM国际基因工程机器大赛和合成生物学领域的学者专家，你将根据历年的研究为同领域的人员提供相关知识和建议。请根据以下上下文信息回答问题：

        {context}

        问题：{query}

        回答要求：
        1. 仅引用与问题强相关的段落，忽略无关内容；
        2. 整合上下文信息，给出专业、结构化回答；
        3. 标注来源（团队+年份）；
        4. 如上下文不足，请明确说明“根据现有信息无法确定”，并可补充通用知识（需标注“通用知识”）。
        """

    def call_qwen_api(self, prompt: str) -> str:
        """调用 Qwen API"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.5,
                top_p=0.8,
                presence_penalty=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API 调用失败: {str(e)}")

    def postprocess_answer(self, answer: str) -> str:
        """后处理答案"""
        processed = f"## iGEM专家回答\n\n{answer}"
        if "【来源说明】" not in answer:
            processed += "\n\n---\n*注：以上回答基于检索到的 iGEM 项目信息*"
        return processed

    def query(self, question: str) -> Dict:
        """端到端 RAG 流程"""
        chunks = self.hybrid_retrieve(question, k=8)
        prompt = self.generate_prompt(question, chunks)
        answer = self.call_qwen_api(prompt)
        processed_answer = self.postprocess_answer(answer)
        return {
            "question": question,
            "answer": processed_answer,
            "sources": [c["metadata"] for c in chunks]
        }
