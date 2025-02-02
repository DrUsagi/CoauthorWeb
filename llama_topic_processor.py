import json
import pickle
from typing import List, Dict, Set
import requests
import time
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class LlamaTopicProcessor:
    def __init__(self, max_topics: int = 200):
        self.max_topics = max_topics
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_embeddings = {}
        self.topic_clusters = {}
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def generate_llama_response(self, prompt: str) -> str:
        """调用本地Ollama服务获取响应"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"API响应错误，状态码: {response.status_code}")
                return ""
            
            # 只获取response字段的内容
            if 'response' in response.json():
                return response.json()['response']
            else:
                print(f"响应中没有找到 'response' 字段")
                return ""
            
        except requests.exceptions.ConnectionError:
            print(f"无法连接到Ollama服务，请确保服务已启动（localhost:11434）")
            print("请运行: ollama run llama3.2:3b")
            return ""
        except requests.exceptions.Timeout:
            print(f"请求超时")
            return ""
        except Exception as e:
            print(f"调用Ollama API时出错: {str(e)}")
            return ""

    def extract_topics_from_paper(self, title: str, abstract: str) -> List[str]:
        """使用LLM从单篇论文中提取主题"""
        prompt = f"""As an academic expert, please analyze the following paper's title and abstract to extract 3-5 main research topics.
Each topic should be a short phrase (2-4 words) describing the paper's key research areas, methods, or applications.
Please respond in English only.

Title: {title}
Abstract: {abstract}

Topics:"""

        response = self.generate_llama_response(prompt)
        
        # 处理响应文本，提取主题
        topics = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                topic = line.split('. ', 1)[1].strip('[]')
                if 2 <= len(topic.split()) <= 4:  # 只保留2-4个词的主题
                    topics.append(topic.lower())
        
        return topics

    def summarize_all_topics(self, all_topics: Set[str]) -> List[str]:
        """使用LLM总结和归纳主题"""
        if not all_topics:
            print("没有找到任何主题可供总结")
            return []
        
        # 将主题列表转换为字符串
        topics_text = "\n".join([f"- {topic}" for topic in sorted(all_topics)])
        
        prompt = f"""As an academic expert, please analyze and summarize the following research topics into {self.max_topics} representative topics.
Each summary topic should be a concise phrase (2-4 words) that captures a key research area.
Please respond in English only.

Original topics:
{topics_text}

Please provide a list of representative summary topics in the following format:
1. [Summary Topic 1]
2. [Summary Topic 2]
...

Only list the topics, no additional explanation needed."""

        response = self.generate_llama_response(prompt)
        
        # 处理响应文本，提取总结后的主题
        summarized_topics = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                topic = line.split('. ', 1)[1].strip('[]')
                if 2 <= len(topic.split()) <= 4:  # 只保留2-4个词的主题
                    summarized_topics.append(topic.lower())
                
        # 添加调试信息
        print(f"原始主题数量: {len(all_topics)}")
        print(f"总结后主题数量: {len(summarized_topics)}")
        if not summarized_topics:
            print("警告：总结后没有得到任何主题")
            print("LLM响应内容:")
            print(response)
        
        return summarized_topics[:self.max_topics]  # 确保不超过最大主题数

    def cluster_topics(self, all_topics: Set[str], summarized_topics: List[str]) -> Dict[str, str]:
        """使用BERT嵌入将原始主题映射到总结后的主题"""
        print("开始对主题进行嵌入...")
        
        # 获取所有主题的嵌入
        all_embeddings = {}
        summarized_embeddings = {}
        
        for topic in all_topics:
            all_embeddings[topic] = self.bert_model.encode([topic])[0]
        
        for topic in summarized_topics:
            summarized_embeddings[topic] = self.bert_model.encode([topic])[0]
        
        # 计算相似度并分配主题
        topic_clusters = {}
        for orig_topic in all_topics:
            orig_embedding = all_embeddings[orig_topic]
            max_sim = -1
            best_match = None
            
            for summ_topic in summarized_topics:
                summ_embedding = summarized_embeddings[summ_topic]
                sim = cosine_similarity([orig_embedding], [summ_embedding])[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = summ_topic
            
            topic_clusters[orig_topic] = best_match
        
        return topic_clusters

    def process_papers(self, json_file: str) -> Dict:
        """处理论文数据，提取并聚类主题"""
        print("开始处理论文数据...")
        
        # 首先测试Ollama连接
        test_prompt = "Hello, are you ready?"
        test_response = self.generate_llama_response(test_prompt)
        if not test_response:
            print("无法连接到Ollama服务，请检查服务是否正常运行")
            print("请确保已经运行了以下命令：")
            print("ollama run llama3.2:3b")
            return {}
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                print(f"正在读取 {json_file}...")
                data = json.load(f)
                papers = data.get('papers', {})
                print(f"找到 {len(papers)} 篇论文")
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            raise
        
        # 收集所有主题
        all_topics = set()
        paper_topics = {}
        paper_info = {}
        
        # 使用tqdm创建进度条
        for paper_id, info in tqdm(papers.items(), desc="处理论文", total=len(papers)):
            title = info.get('title', '')
            abstract = info.get('abstract', '')
            authors = info.get('authors', [])
            
            if isinstance(authors, str):
                authors = [authors]
            
            topics = self.extract_topics_from_paper(title, abstract)
            if topics:
                paper_topics[paper_id] = topics
                all_topics.update(topics)
                
                paper_info[paper_id] = {
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'topics': topics
                }
            
            # 每处理10篇论文暂停一下，避免请求太频繁
            if len(paper_topics) % 10 == 0:
                time.sleep(1)
        
        print(f"\n提取完成，共找到 {len(all_topics)} 个独特主题")
        print("开始总结主题...")
        
        # 使用LLM总结主题
        summarized_topics = self.summarize_all_topics(all_topics)
        print(f"总结完成，得到 {len(summarized_topics)} 个代表性主题")
        
        # 将原始主题映射到总结后的主题
        topic_clusters = self.cluster_topics(all_topics, summarized_topics)
        
        # 保存处理结果
        results = {
            'topic_clusters': topic_clusters,
            'paper_info': paper_info,
            'paper_topics': paper_topics,
            'summarized_topics': summarized_topics
        }
        
        print("正在保存处理结果...")
        with open('llama_processed_topics.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("处理完成！")
        return results

def main():
    processor = LlamaTopicProcessor(max_topics=150)  # 设置最大主题数为150
    try:
        results = processor.process_papers('papers_db.json')
        
        # 打印统计信息
        print("\n主题聚类结果统计：")
        print(f"原始主题数量: {len(results['paper_topics'])}")
        print(f"总结后的主题数量: {len(results['summarized_topics'])}")
        
        # 打印一些主题示例
        print("\n主题示例：")
        for topic in list(results['summarized_topics'])[:10]:
            related = [k for k, v in results['topic_clusters'].items() if v == topic]
            print(f"\n主题: {topic}")
            print(f"相关主题: {', '.join(related[:5])}...")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 