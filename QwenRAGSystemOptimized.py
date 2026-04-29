# ================ github Streamlit 部署 ===========
import re
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

# 新增：语言检测
try:
    from langdetect import detect
except ImportError:
    raise ImportError("请安装 langdetect: pip install langdetect")



class QwenRAGSystemOptimized:
    def __init__(self, api_key: str = None):
        # ========== API Key ==========
        """
        api_key: 用户可手动传入 Qwen API Key，如果不传则读取环境变量 DASHSCOPE_API_KEY
        """
        load_dotenv()

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found. Please set Qwen's API key as an environment variable."
            )

        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )

        # ========== 索引和元数据 ==========
        if not Path("./out_embedding/vector_index.faiss").exists():
            raise FileNotFoundError("缺少 vector_index.faiss，请检查路径")
        if not Path("./out_embedding/chunk_metadata.json").exists():
            raise FileNotFoundError("缺少 chunk_metadata.json，请检查路径")

        self.index = faiss.read_index("./out_embedding/vector_index.faiss")
        with open("./out_embedding/chunk_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # ========== 嵌入模型 ==========
        self.embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

        # ========== 交叉编码器 ==========
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

        # ========== 初始化 BM25 ==========
        self._prepare_bm25_corpus()

    def _extract_hard_filters(self, query: str) -> dict:
        """从用户提问中提取强过滤条件（如年份）"""
        filters = {}
        # 匹配 2000-2099 之间的4位数字作为年份
        year_match = re.search(r'(20\d{2})', query)
        if year_match:
            filters['year'] = int(year_match.group(1))
        return filters

    def _prepare_bm25_corpus(self):
        """预处理BM25检索语料"""
        if not self.metadata:
            raise ValueError("metadata 为空，无法初始化 BM25")

        self.all_texts = [item["text"] for item in self.metadata]
        # 使用 jieba 进行分词
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.all_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # def hybrid_retrieve(self, query: str, k: int = 10, bm25_candidates: int = 100) -> List[Dict]:
    #     """混合检索：BM25初筛 + 向量检索 + 交叉编码器重排序"""
    #     if not hasattr(self, "bm25"):
    #         raise RuntimeError("BM25 尚未初始化，请检查 _prepare_bm25_corpus()")

    #     # 1. BM25 初筛
    #     tokenized_query = list(jieba.cut(query))
    #     bm25_scores = self.bm25.get_scores(tokenized_query)
    #     top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_candidates]

    #     # 2. 向量检索
    #     query_embed = self.embed_model.encode([query])
    #     candidate_texts = [self.all_texts[i] for i in top_bm25_indices]
    #     candidate_embeddings = self.embed_model.encode(candidate_texts)
    #     vector_scores = cosine_similarity(query_embed, candidate_embeddings)[0]

    #     # 3. 融合 BM25 和向量得分
    #     bm25_normalized = (bm25_scores[top_bm25_indices] - np.min(bm25_scores[top_bm25_indices])) / (
    #         np.max(bm25_scores[top_bm25_indices]) - np.min(bm25_scores[top_bm25_indices]) + 1e-8
    #     )
    #     vector_normalized = (vector_scores - np.min(vector_scores)) / (
    #         np.max(vector_scores) - np.min(vector_scores) + 1e-8
    #     )

    #     hybrid_scores = 0.3 * bm25_normalized + 0.7 * vector_normalized
    #     top_hybrid_indices = top_bm25_indices[np.argsort(hybrid_scores)[::-1][:k]]

    #     # 4. 交叉编码器重排序
    #     rerank_pairs = [(query, self.all_texts[i]) for i in top_hybrid_indices]
    #     rerank_scores = self._batch_rerank(rerank_pairs)

    #     results = []
    #     for idx, score in zip(top_hybrid_indices, rerank_scores):
    #         chunk_data = {
    #             "text": self.metadata[idx]["text"],
    #             "metadata": self.metadata[idx]["metadata"],
    #             "score": float(score)
    #         }
    #         results.append(chunk_data)

    #     return results

    def hybrid_retrieve(self, query: str, k: int = 10, recall_candidates: int = 50) -> List[Dict]:
        """【真·双线并行】混合检索：BM25召回 + FAISS向量召回 -> 规则拦截 -> 交叉编码器重排序"""
        if not hasattr(self, "bm25"):
            raise RuntimeError("BM25 尚未初始化，请检查 _prepare_bm25_corpus()")

        # 1. 提前提取问题中的硬性条件（如年份）
        filters = self._extract_hard_filters(query)

        # ==================== 并行召回阶段 ====================
        
        # 【左手】：BM25 稀疏召回 (专注字面关键字匹配)
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 取字面分数最高的 Top N 个
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:recall_candidates].tolist()

        # 【右手】：FAISS 向量召回 (专注深层语义理解)
        # 把问题变成向量 (注意：FAISS 强制要求 float32 格式)
        query_embed = self.embed_model.encode([query]).astype(np.float32)
        # 直接去你之前建好的 FAISS 库里秒查！
        faiss_distances, faiss_indices_2d = self.index.search(query_embed, recall_candidates)
        top_faiss_indices = faiss_indices_2d[0].tolist()

        # ==================== 融合与过滤阶段 ====================

        # 【合并候选池并去重】：把两边找出来的人混在一起
        # 使用 set 去重，可能总共会有 50 ~ 100 个唯一的候选人
        combined_indices = list(set(top_bm25_indices + top_faiss_indices))

        # 【规则后置过滤】：按年份无情剔除
        filtered_indices = []
        for idx in combined_indices:
            # 容错：去除 faiss 有时找不够数量返回的 -1 占位符
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            chunk_meta = self.metadata[idx]["metadata"]
            
            # 如果提问中有年份，且数据块年份不匹配，直接拦截
            if 'year' in filters and chunk_meta.get('year') != filters['year']:
                continue
                
            filtered_indices.append(idx)
            
        # 容错：如果过滤条件太严导致全军覆没，降级回退到过滤前的结果以防程序崩溃
        if not filtered_indices:
            filtered_indices = combined_indices[:k*2]

        # ==================== 精排面试阶段 ====================

        # 【终极面试：交叉编码器重排序】
        # 让最聪明的 Cross-Encoder 对剩下的候选人逐一打分
        rerank_pairs = [(query, self.all_texts[i]) for i in filtered_indices]
        rerank_scores = self._batch_rerank(rerank_pairs)

        # 组装结果
        results = []
        for idx, score in zip(filtered_indices, rerank_scores):
            results.append({
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx]["metadata"],
                "score": float(score)
            })
            
        # 根据 Cross-Encoder 得分从高到低排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 最终只交出最顶尖的 k 个作业
        return results[:k]

    def flexible_retrieve(self, query: str, k: int = 10, recall_candidates: int = 50,
                          use_bm25: bool = True, use_faiss: bool = True,
                          use_reranker: bool = True, use_year_filter: bool = True) -> List[Dict]:
        """可配置检索，用于消融实验。无 Reranker 时使用 RRF 融合排序。"""
        if not use_bm25 and not use_faiss:
            raise ValueError("use_bm25 和 use_faiss 不能同时为 False")

        filters = self._extract_hard_filters(query) if use_year_filter else {}

        bm25_ordered: list = []
        faiss_ordered: list = []
        raw_bm25_scores = None

        if use_bm25:
            tokenized_query = list(jieba.cut(query))
            raw_bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_ordered = np.argsort(raw_bm25_scores)[::-1][:recall_candidates].tolist()

        if use_faiss:
            query_embed = self.embed_model.encode([query]).astype(np.float32)
            faiss_distances, faiss_idx_2d = self.index.search(query_embed, recall_candidates)
            faiss_ordered = [i for i in faiss_idx_2d[0].tolist() if i >= 0]

        combined = list(set(bm25_ordered + faiss_ordered))

        yr = filters.get('year')
        if yr:
            filtered = [i for i in combined
                        if 0 <= i < len(self.metadata)
                        and self.metadata[i]['metadata'].get('year') == yr]
            if not filtered:
                filtered = [i for i in combined if 0 <= i < len(self.metadata)]
        else:
            filtered = [i for i in combined if 0 <= i < len(self.metadata)]

        if not filtered:
            return []

        if use_reranker:
            rerank_pairs = [(query, self.all_texts[i]) for i in filtered]
            scores = self._batch_rerank(rerank_pairs)
            results = [{"text": self.metadata[i]["text"],
                        "metadata": self.metadata[i]["metadata"],
                        "score": float(s)}
                       for i, s in zip(filtered, scores)]
        else:
            RRF_K = 60
            bm25_rank = {idx: r + 1 for r, idx in enumerate(bm25_ordered)}
            faiss_rank = {idx: r + 1 for r, idx in enumerate(faiss_ordered)}
            results = []
            for idx in filtered:
                if use_bm25 and use_faiss:
                    r1 = bm25_rank.get(idx, recall_candidates + 1)
                    r2 = faiss_rank.get(idx, recall_candidates + 1)
                    score = 1.0 / (RRF_K + r1) + 1.0 / (RRF_K + r2)
                elif use_bm25:
                    r1 = bm25_rank.get(idx, recall_candidates + 1)
                    score = 1.0 / (RRF_K + r1)
                else:
                    r2 = faiss_rank.get(idx, recall_candidates + 1)
                    score = 1.0 / (RRF_K + r2)
                results.append({"text": self.metadata[idx]["text"],
                                "metadata": self.metadata[idx]["metadata"],
                                "score": score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def _batch_rerank(self, rerank_pairs: List[tuple]) -> List[float]:
        """批量处理交叉编码器以加速重排序"""
        with ThreadPoolExecutor() as executor:
            rerank_scores = list(executor.map(self.cross_encoder.predict, rerank_pairs))
        return rerank_scores

    def generate_prompt(self, query: str, chunks: List[Dict]) -> str:
        """构建增强提示（始终用中文）"""
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
                # model="qwen-plus",
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.5,
                top_p=0.8,
                presence_penalty=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API 调用失败: {str(e)}")

    def _translate_with_qwen(self, text: str, target_lang: str) -> str:
        """使用 Qwen 模型进行翻译"""
        if target_lang == "zh":
            instruction = "请将以下英文翻译成专业、简洁的中文："
        elif target_lang == "en":
            instruction = "Please translate the following Chinese into professional, concise English:"
        else:
            return text

        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": f"{instruction}\n\n{text}"}],
                max_tokens=1000,
                temperature=0.3,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # 如果翻译失败，返回原文（避免中断）
            print(f"Translation failed: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """检测文本语言，返回 'zh' 或 'en'"""
        try:
            lang = detect(text)
            if lang.startswith("zh"):
                return "zh"
            elif lang == "en":
                return "en"
            else:
                # 默认中文（因为文档是中文）
                return "zh"
        except:
            return "zh"

    def postprocess_answer(self, answer: str) -> str:
        """后处理答案"""
        processed = f"## iGEM Expert Answer\n\n{answer}"
        if "【Source】" not in answer and "来源" not in answer:
            processed += "\n\n---\n*Note: This response is based on retrieved iGEM project information.*"
        return processed

    def query(self, question: str) -> Dict:
        """端到端 RAG 流程，支持中英文输入"""
        # Step 1: 检测用户问题语言
        user_lang = self.detect_language(question)
        is_english_input = (user_lang == "en")

        # Step 2: 如果是英文，翻译成中文用于 RAG
        query_for_rag = self._translate_with_qwen(question, "zh") if is_english_input else question

        # Step 3: 执行 RAG（内部全程中文）
        chunks = self.hybrid_retrieve(query_for_rag, k=8)
        prompt = self.generate_prompt(query_for_rag, chunks)
        chinese_answer = self.call_qwen_api(prompt)

        # Step 4: 如果用户用英文提问，将答案翻译回英文
        final_answer = self._translate_with_qwen(chinese_answer, "en") if is_english_input else chinese_answer

        # Step 5: 后处理（根据语言调整提示语）
        processed_answer = self.postprocess_answer(final_answer)

        return {
            "question": question,
            "answer": processed_answer,
            "sources": [c["metadata"] for c in chunks]
        }
