import json
import networkx as nx
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import argparse

class AcademicNetwork:
    def __init__(self, data_source='llama'):
        """
        初始化网络分析器
        """
        self.topic_graph = nx.Graph()  # 主题关系图
        self.author_graphs = defaultdict(nx.Graph)  # 每个主题的作者关系图
        self.author_papers = defaultdict(list)  # 存储作者和他们的论文
        self.topic_papers = defaultdict(list)  # 存储主题和相关论文
        self.topic_authors = defaultdict(set)  # 存储主题和相关作者
        
        # 加载数据文件
        data_file = 'llama_processed_topics.pkl'
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到数据文件 {data_file}，请先运行 llama_topic_processor.py 生成数据")
            
        # 加载处理好的数据
        with open(data_file, 'rb') as f:
            self.processed_data = pickle.load(f)
        
        self.topic_clusters = self.processed_data['topic_clusters']

    def get_embedding(self, text):
        """获取文本的BERT嵌入"""
        return self.bert_model.encode([text])[0]
    
    def cluster_similar_topics(self, topics, threshold=0.8):
        """使用BERT嵌入和余弦相似度聚类相似主题"""
        if not topics:
            return {}
        
        # 获取所有主题的嵌入
        topic_embeddings = {topic: self.get_embedding(topic) for topic in topics}
        self.topic_embeddings = topic_embeddings
        
        # 计算相似度矩阵
        embeddings = np.array(list(topic_embeddings.values()))
        similarity_matrix = cosine_similarity(embeddings)
        
        # 创建一个字典来存储主题聚类结果
        topic_clusters = {}
        topics_list = list(topics)
        processed = set()
        
        # 基于相似度阈值直接聚类
        for i, topic1 in enumerate(topics_list):
            if topic1 in processed:
                continue
            
            # 如果这个主题还没有被处理，它将成为一个新簇的代表
            cluster_representative = topic1
            topic_clusters[topic1] = cluster_representative
            processed.add(topic1)
            
            # 查找与该主题相似的其他主题
            for j, topic2 in enumerate(topics_list[i+1:], i+1):
                if topic2 in processed:
                    continue
                
                # 如果两个主题的相似度超过阈值，将它们归为同一簇
                if similarity_matrix[i, j] > threshold:
                    topic_clusters[topic2] = cluster_representative
                    processed.add(topic2)
        
        return topic_clusters

    def extract_keywords(self, text, top_n=5):
        """从文本中提取关键词"""
        # 清理文本
        text = re.sub(r'[^\w\s]', '', text)
        words = text.lower().split()
        # 移除停用词
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words and len(w) > 2]
        # 返回最常见的词
        freq_dist = nltk.FreqDist(words)
        return [word for word, _ in freq_dist.most_common(top_n)]

    def load_data(self):
        """构建网络关系"""
        paper_info = self.processed_data['paper_info']
        
        # 使用聚类后的主题构建网络
        for paper_id, info in paper_info.items():
            topics = info['topics']  # llama处理器使用'topics'
            
            clustered_keywords = [self.topic_clusters[kw] for kw in topics]
            clustered_keywords = list(set(clustered_keywords))
            
            # 更新主题关系图
            for i, kw1 in enumerate(clustered_keywords):
                self.topic_graph.add_node(kw1)
                for kw2 in clustered_keywords[i+1:]:
                    if self.topic_graph.has_edge(kw1, kw2):
                        self.topic_graph[kw1][kw2]['weight'] += 1
                    else:
                        self.topic_graph.add_edge(kw1, kw2, weight=1)
                
                # 更新主题-论文关系
                self.topic_papers[kw1].append({
                    'title': info['title'],
                    'abstract': info['abstract'],
                    'authors': info['authors']
                })
                
                # 更新主题-作者关系
                self.topic_authors[kw1].update(info['authors'])
                
                # 更新该主题下的作者合作网络
                author_graph = self.author_graphs[kw1]
                for author in info['authors']:
                    author_graph.add_node(author)
                
                for i in range(len(info['authors'])):
                    for j in range(i + 1, len(info['authors'])):
                        if author_graph.has_edge(info['authors'][i], info['authors'][j]):
                            author_graph[info['authors'][i]][info['authors'][j]]['weight'] += 1
                        else:
                            author_graph.add_edge(info['authors'][i], info['authors'][j], weight=1)

    def create_topic_network(self):
        """创建主题关系的交互式网络图"""
        # 增加k值使节点分布更分散，增加iterations使布局更稳定
        pos = nx.spring_layout(self.topic_graph, k=3, iterations=100)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.topic_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            papers_count = len(self.topic_papers[node])
            authors_count = len(self.topic_authors[node])
            original_keywords = [k for k, v in self.topic_clusters.items() if v == node]
            
            # 缩短显示的关键词数量并将首字母大写
            if len(original_keywords) > 5:
                original_keywords = original_keywords[:5] + ['...']
            
            # 将关键词首字母大写
            original_keywords = [k.title() for k in original_keywords]
            # 将主题首字母大写
            node_title = node.title()
            
            node_text.append(f"主题: {node_title}<br>"
                           f"关键词: {', '.join(original_keywords)}<br>"
                           f"论文数: {papers_count}<br>"
                           f"作者数: {authors_count}")
            
            # 调整节点大小计算方式，使用对数缩放
            node_size.append(np.log2(papers_count + 1) * 15)

        # 准备边的数据
        edge_x = []
        edge_y = []
        for edge in self.topic_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # 创建边的跟踪对象，增加透明度
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(169,169,169,0.3)'),
            hoverinfo='none',
            mode='lines')

        # 设置节点颜色基于连接数
        node_adjacencies = []
        for node in self.topic_graph.nodes():
            node_adjacencies.append(len(list(self.topic_graph.neighbors(node))))

        # 优化节点的视觉效果
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n.title() for n in self.topic_graph.nodes()],  # 将节点标签首字母大写
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                showscale=True,
                size=node_size,
                colorscale='Viridis',
                reversescale=True,
                color=node_adjacencies,
                line=dict(width=1, color='rgba(255,255,255,0.5)'),
                opacity=0.8
            ),
            customdata=list(self.topic_graph.nodes())
        )

        # 设置布局
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text='研究主题关系网络',
                            font=dict(size=20, color='#444'),
                            x=0.5,
                            y=0.95
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        plot_bgcolor='rgba(255,255,255,0.9)',  # 略微灰色背景
                        paper_bgcolor='white',
                        xaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            scaleanchor="y",
                            scaleratio=1,
                            range=[-3, 3]  # 扩大显示范围
                        ),
                        yaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            range=[-3, 3]  # 扩大显示范围
                        ),
                        height=900,  # 增加高度
                        width=None,
                        autosize=True,
                        dragmode='pan'
                        )
                    )

        # 添加水印和说明
        fig.add_annotation(
            text="点击节点查看详情",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12, color="gray"),
            opacity=0.7
        )

        return fig

    def create_author_network(self, topic):
        """创建特定主题下作者关系的交互式网络图"""
        if topic not in self.author_graphs:
            return None
            
        author_graph = self.author_graphs[topic]
        pos = nx.spring_layout(author_graph)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for author in author_graph.nodes():
            x, y = pos[author]
            node_x.append(x)
            node_y.append(y)
            
            # 获取该作者在这个主题下的所有论文
            author_papers = [
                paper['title'] 
                for paper in self.topic_papers[topic] 
                if author in paper['authors']
            ]
            paper_count = len(author_papers)
            
            # 构建悬停文本，包含作者名、论文数和论文标题列表
            hover_text = [
                f"作者: {author}",
                f"论文数: {paper_count}",
                "\n论文列表:"
            ]
            # 添加每篇论文的标题，限制长度并编号
            for i, title in enumerate(author_papers, 1):
                if len(title) > 100:  # 如果标题太长，截断它
                    title = title[:97] + "..."
                hover_text.append(f"{i}. {title}")
            
            node_text.append("<br>".join(hover_text))
            node_size.append(paper_count * 20)

        # 准备边数据
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in author_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 计算两个作者的合作论文
            coauthor_papers = [
                paper['title']
                for paper in self.topic_papers[topic]
                if edge[0] in paper['authors'] and edge[1] in paper['authors']
            ]
            edge_text.append(f"合作论文数: {len(coauthor_papers)}")

        # 创建边的跟踪对象
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            hovertext=edge_text,
            mode='lines')

        # 修改节点的跟踪对象
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"{author}<br>({sum(1 for paper in self.topic_papers[topic] if author in paper['authors'])}篇)" 
                  for author in author_graph.nodes()],  # 显示作者名和论文数
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                size=node_size,
                colorscale='Viridis',
                line_width=2),
            customdata=['back'] * len(node_x)
        )

        # 创建图形并添加返回按钮
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text=f'主题 "{topic.title()}" 下的作者合作网络',  # 移除返回按钮提示
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=900
                        ))
        
        return fig

    def print_topic_stats(self, topic):
        """打印特定主题的统计信息"""
        if topic not in self.topic_papers:
            print(f"未找到主题 '{topic}' 的相关信息")
            return
            
        papers = self.topic_papers[topic]
        authors = self.topic_authors[topic]
        
        print(f"\n主题 '{topic}' 统计信息:")
        print(f"相关论文数量: {len(papers)}")
        print(f"相关作者数量: {len(authors)}")
        print("\n相关论文:")
        for paper in papers:
            print(f"- {paper['title']}")

def main():
    # 移除命令行参数，直接使用llama数据源
    try:
        network = AcademicNetwork()
        network.load_data()
    except FileNotFoundError as e:
        print(e)
        print("\n请先运行 llama_topic_processor.py 生成数据")
        return
        
    import dash
    from dash import html, dcc
    from dash.dependencies import Input, Output, State
    
    # 创建Dash应用
    app = dash.Dash(__name__)
    
    # 设置应用布局
    app.layout = html.Div([
        html.Div([
            html.H1("学术主题网络分析",
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            dcc.Graph(
                id='network-graph',
                style={'height': '90vh'}
            ),
            html.Button(
                '返回主题网络', 
                id='back-button', 
                style={
                    'display': 'none',
                    'position': 'fixed',
                    'bottom': '20px',
                    'left': '20px',
                    'zIndex': '1000',
                    'padding': '10px 20px',
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                    'fontSize': '14px'
                }
            )
        ], style={
            'position': 'relative',
            'height': '100vh',
            'width': '100%',
            'padding': '10px'
        })
    ])
    
    # 存储当前显示的图形类型
    current_view = {'type': 'topic'}
    
    @app.callback(
        Output('network-graph', 'figure'),
        [Input('network-graph', 'clickData'),
         Input('back-button', 'n_clicks')],
        [State('network-graph', 'figure')],
        prevent_initial_call=False
    )
    def update_graph(clickData, n_clicks, current_figure):
        ctx = dash.callback_context
        if not ctx.triggered:
            return network.create_topic_network()
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'back-button':
            current_view['type'] = 'topic'
            return network.create_topic_network()
        elif trigger_id == 'network-graph' and current_view['type'] == 'topic':
            try:
                topic = clickData['points'][0]['customdata']
                current_view['type'] = 'author'
                return network.create_author_network(topic)
            except (KeyError, TypeError, IndexError):
                return current_figure
        
        return current_figure
    
    @app.callback(
        Output('back-button', 'style'),
        [Input('network-graph', 'clickData'),
         Input('back-button', 'n_clicks')],
        [State('back-button', 'style')],
        prevent_initial_call=False
    )
    def toggle_button(clickData, n_clicks, current_style):
        ctx = dash.callback_context
        if not ctx.triggered:
            return {'display': 'none'}
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'network-graph' and current_view['type'] == 'topic':
            try:
                if clickData['points'][0]['customdata']:
                    return {
                        'display': 'block',
                        'position': 'fixed',
                        'bottom': '20px',
                        'left': '20px',
                        'zIndex': '1000',
                        'padding': '10px 20px',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                        'fontSize': '14px'
                    }
            except (KeyError, TypeError, IndexError):
                pass
        elif trigger_id == 'back-button':
            return {'display': 'none'}
            
        return current_style or {'display': 'none'}

    # 添加CSS样式
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>学术网络分析</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # 运行应用
    print("\n使用 llama 数据源启动应用...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main() 