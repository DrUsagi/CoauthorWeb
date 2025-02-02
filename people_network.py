import json
import networkx as nx
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import pickle
import os
import argparse
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px  # 用于颜色集

class PeopleNetwork:
    def __init__(self, data_source='llama'):
        """
        初始化人物网络分析器
        """
        self.people_graph = nx.Graph()  # 全局作者关系图
        self.topic_graphs = defaultdict(nx.Graph)  # 每个作者的主题关系图
        self.author_papers = defaultdict(list)  # 存储作者和他们的论文
        self.author_topics = defaultdict(set)  # 存储作者和他们的研究主题
        self.topic_collaborators = defaultdict(lambda: defaultdict(set))  # 存储每个主题下的合作者
        
        # 加载数据文件
        data_file = 'llama_processed_topics.pkl'
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到数据文件 {data_file}，请先运行 llama_topic_processor.py 生成数据")
            
        # 加载处理好的数据
        with open(data_file, 'rb') as f:
            self.processed_data = pickle.load(f)
        
        self.topic_clusters = self.processed_data['topic_clusters']

    def load_data(self):
        """构建网络关系"""
        paper_info = self.processed_data['paper_info']
        
        # 构建作者关系网络
        for paper_id, info in paper_info.items():
            authors = info['authors']
            topics = info['topics']  # llama处理器使用'topics'
            
            # 更新作者的论文列表
            for author in authors:
                self.author_papers[author].append({
                    'title': info['title'],
                    'abstract': info['abstract'],
                    'topics': topics,
                    'authors': authors
                })
                
                # 更新作者的主题
                clustered_topics = {self.topic_clusters[t] for t in topics}
                self.author_topics[author].update(clustered_topics)
                
                # 更新主题下的合作关系
                for topic in clustered_topics:
                    for other_author in authors:
                        if other_author != author:
                            self.topic_collaborators[author][topic].add(other_author)
            
            # 更新全局作者关系网络
            for i, author1 in enumerate(authors):
                self.people_graph.add_node(author1)
                for author2 in authors[i+1:]:
                    if self.people_graph.has_edge(author1, author2):
                        self.people_graph[author1][author2]['weight'] += 1
                    else:
                        self.people_graph.add_edge(author1, author2, weight=1)

    def create_people_network(self):
        """创建作者关系的交互式网络图"""
        # 只保留论文数量大于等于3的作者
        significant_authors = {
            author for author in self.people_graph.nodes()
            if len(self.author_papers[author]) >= 3
        }
        
        # 创建子图，只包含重要作者
        subgraph = self.people_graph.subgraph(significant_authors)
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for author in subgraph.nodes():
            x, y = pos[author]
            node_x.append(x)
            node_y.append(y)
            
            papers = self.author_papers[author]
            topics = self.author_topics[author]
            
            # 构建悬停文本
            hover_text = [
                f"作者: {author}",
                f"论文数: {len(papers)}",
                f"研究主题数: {len(topics)}",
                "\n主要研究主题:",
            ]
            for topic in topics:
                hover_text.append(f"- {topic.title()}")
            
            node_text.append("<br>".join(hover_text))
            # 使用论文数量确定节点大小
            node_size.append(np.sqrt(len(papers)) * 20)

        # 准备边数据
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = self.people_graph[edge[0]][edge[1]]['weight']
            edge_text.append(f"合作论文数: {weight}")

        # 创建边的跟踪对象
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(169,169,169,0.3)'),
            hoverinfo='text',
            hovertext=edge_text,
            mode='lines')

        # 创建节点的跟踪对象
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"{author}<br>({len(self.author_papers[author])}篇)" 
                  for author in subgraph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                showscale=False,
                size=node_size,
                color='lightblue',
                line=dict(width=1, color='rgba(255,255,255,0.5)'),
                opacity=0.8
            ),
            customdata=list(subgraph.nodes())
        )

        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text='研究者合作网络 (≥3篇论文)',
                            font=dict(size=20, color='#444'),
                            x=0.5,
                            y=0.95
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=900,
                        width=None,
                        autosize=True
                        ))

        return fig

    def create_author_topic_network(self, author):
        """创建特定作者的主题合作网络"""
        if author not in self.author_topics:
            return None
            
        # 创建这个作者的主题-合作者网络
        G = nx.Graph()
        
        # 为不同主题生成不同的颜色
        topics = list(self.author_topics[author])
        colors = px.colors.qualitative.Set3[:len(topics)]
        topic_colors = dict(zip(topics, colors))
        
        # 创建一个字典来存储每个主题下的论文
        topic_papers = defaultdict(set)  # 使用set避免重复
        for paper in self.author_papers[author]:
            for topic in paper['topics']:
                clustered_topic = self.topic_clusters[topic]
                if clustered_topic in topic_colors:
                    topic_papers[clustered_topic].add(paper['title'])
        
        # 创建一个字典来存储与每个合作者的合作论文
        collaborator_papers = defaultdict(lambda: defaultdict(set))  # 使用嵌套的set避免重复
        
        # 添加主题节点和合作关系
        for topic in topics:
            G.add_node(topic, type='topic', color=topic_colors[topic])
            # 添加在该主题下的合作者
            for paper in self.author_papers[author]:
                if any(self.topic_clusters[t] == topic for t in paper['topics']):
                    for coauthor in paper['authors']:
                        if coauthor != author:
                            G.add_node(coauthor, type='collaborator')
                            G.add_edge(topic, coauthor, color=topic_colors[topic])
                            collaborator_papers[coauthor][topic].add(paper['title'])
        
        pos = nx.spring_layout(G, k=2)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if G.nodes[node]['type'] == 'topic':
                # 主题节点
                papers_in_topic = topic_papers[node]
                hover_text = [
                    f"主题: {node.title()}",
                    f"论文数: {len(papers_in_topic)}",
                    "\n该主题下的论文:",
                ]
                # 添加该主题下的所有论文标题
                for i, title in enumerate(sorted(papers_in_topic), 1):
                    if len(title) > 100:
                        title = title[:97] + "..."
                    hover_text.append(f"{i}. {title}")
                
                node_text.append("<br>".join(hover_text))
                node_size.append(len(papers_in_topic) * 20)
                node_color.append(G.nodes[node]['color'])
            else:
                # 合作者节点
                coauthor_papers = set()  # 使用set避免重复
                topic_colors_for_author = set()
                
                # 收集所有主题下与该合作者的合作论文
                for topic, papers in collaborator_papers[node].items():
                    coauthor_papers.update(papers)
                    topic_colors_for_author.add(topic_colors[topic])
                
                hover_text = [
                    f"合作者: {node}",
                    f"合作论文数: {len(coauthor_papers)}",  # 使用去重后的数量
                    "\n合作论文:",
                ]
                for i, title in enumerate(sorted(coauthor_papers), 1):  # 对论文标题排序
                    if len(title) > 100:
                        title = title[:97] + "..."
                    hover_text.append(f"{i}. {title}")
                
                node_text.append("<br>".join(hover_text))
                node_size.append(len(coauthor_papers) * 15)
                # 如果作者在多个主题下都有合作，使用混合色
                if len(topic_colors_for_author) > 1:
                    node_color.append('lightgray')
                else:
                    node_color.append(list(topic_colors_for_author)[0] if topic_colors_for_author else 'lightgray')

        # 准备边数据，为每条边设置对应主题的颜色
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_color = G.edges[edge]['color']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color=edge_color),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # 创建节点的跟踪对象
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node).title() for node in G.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2),
            customdata=[str(node) for node in G.nodes()]
        )

        # 创建图形
        fig = go.Figure(data=[*edge_traces, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text=f'{author} 的研究主题与合作关系网络',
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

def main():
    # 移除命令行参数，直接使用llama数据源
    try:
        network = PeopleNetwork()
        network.load_data()
    except FileNotFoundError as e:
        print(e)
        print("\n请先运行 llama_topic_processor.py 生成数据")
        return
        
    # 创建Dash应用
    app = dash.Dash(__name__)
    
    # 设置应用布局
    app.layout = html.Div([
        html.Div([
            html.H1("研究者网络分析",
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            dcc.Graph(
                id='network-graph',
                style={'height': '90vh'}
            ),
            html.Button(
                '返回作者网络', 
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
    current_view = {'type': 'people'}
    
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
            return network.create_people_network()
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'back-button':
            current_view['type'] = 'people'
            return network.create_people_network()
        elif trigger_id == 'network-graph':
            try:
                point = clickData['points'][0]
                # 检查是否点击的是节点（最后一个trace）
                if point['curveNumber'] == len(current_figure['data']) - 1:
                    if current_view['type'] == 'people':
                        # 从作者网络页面点击作者
                        author = point['customdata']
                        current_view['type'] = 'topic'
                        return network.create_author_topic_network(author)
                    elif current_view['type'] == 'topic':
                        # 从主题网络页面点击合作者
                        clicked_node = point['customdata']
                        # 如果点击的不是主题节点（即点击的是合作者节点）
                        if not any(topic.lower() == clicked_node.lower() for topic in network.topic_clusters.values()):
                            return network.create_author_topic_network(clicked_node)
            except Exception as e:
                print(f"Error in update_graph: {str(e)}")
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
        
        if trigger_id == 'network-graph' and current_view['type'] == 'people':
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
            <title>研究者网络分析</title>
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