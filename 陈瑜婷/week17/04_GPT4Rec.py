import os
import re
import pandas as pd
from collections import defaultdict
from rank_bm25 import BM25Okapi

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
from openai import OpenAI

# 初始化千问客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取数据
print("正在加载数据...")
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding='latin-1')
movies.columns = ["movie_id", "title", "genres"]

# 过滤评分，保留3分及以上的作为正样本
ratings = ratings[ratings['rating'] >= 3].reset_index(drop=True)

# 按用户和时间排序，构建用户历史
ratings = ratings.sort_values(['user_id', 'timestamp'])
user_history = defaultdict(list)

for _, row in ratings.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    user_history[user_id].append(movie_id)

print(f"用户数量: {len(user_history)}")
print(f"电影数量: {len(movies)}")


def generate_search_queries(user_movie_ids, num_queries=5):
    """
    步骤1：使用千问API生成搜索查询
    
    参数:
        user_movie_ids: 用户历史观看的电影ID列表
        num_queries: 要生成的查询数量
    
    返回:
        queries: 生成的搜索查询列表
    """
    # 获取用户历史电影标题
    user_movies = movies[movies['movie_id'].isin(user_movie_ids)]
    movie_titles = user_movies['title'].tolist()[:10]  # 最多取10个电影标题
    
    if len(movie_titles) == 0:
        return []
    
    # 构建提示词
    movie_list = "\n".join([f"- {title}" for title in movie_titles])
    
    prompt = f"""你是一个电影推荐专家。请根据用户历史观看的电影，生成{num_queries}个能够描述用户兴趣的搜索查询词。
这些查询词应该能够捕捉用户的电影偏好，用于搜索和推荐相似类型的电影。

用户历史观看的电影：
{movie_list}

请生成{num_queries}个简洁的搜索查询词，每个查询词应该能够描述用户的兴趣偏好（如电影类型、主题、风格等）。
每个查询词单独一行，只输出查询词，不需要编号或其他说明。

示例格式：
action sci-fi movies
romantic comedy
thriller suspense
drama with deep themes
classic adventure films
"""
    
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # 解析生成的查询
        queries_text = completion.choices[0].message.content.strip()
        queries = []
        for line in queries_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 移除可能的编号（如 "1. " 或 "- "）
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            if line:
                queries.append(line)
        
        # 如果生成的查询数量不足，截取
        queries = queries[:num_queries]
        
        print(f"生成的搜索查询: {queries}")
        return queries
    
    except Exception as e:
        print(f"生成查询时出错: {e}")
        # 如果API调用失败，返回基于电影类型的简单查询
        genres_list = user_movies['genres'].str.split('|').explode().unique()
        queries = [f"{genre} movies" for genre in genres_list[:num_queries]]
        return queries


def tokenize(text):
    """
    对文本进行分词
    对于英文电影标题和类型，使用简单的空格和标点符号分割
    """
    # 转换为小写
    text = text.lower()
    # 使用正则表达式分割：保留字母数字，去除标点符号
    words = re.findall(r'\b\w+\b', text)
    return words


def retrieve_movies_with_bm25(queries, user_movie_ids, top_k=20):
    """
    步骤2：使用BM25从movies.dat中检索相关电影
    
    参数:
        queries: 搜索查询列表
        user_movie_ids: 用户已观看的电影ID列表（用于过滤）
        top_k: 返回前K个推荐
    
    返回:
        recommended_movies: 推荐的电影列表，包含movie_id, title, score
    """
    if len(queries) == 0:
        return []
    
    # 准备电影文档（标题 + 类型）
    movie_texts = []
    movie_ids = []
    
    for _, row in movies.iterrows():
        movie_id = row['movie_id']
        title = str(row['title'])
        genres = str(row['genres'])
        # 组合标题和类型作为文档内容
        doc_text = f"{title} {genres}"
        movie_texts.append(doc_text)
        movie_ids.append(movie_id)
    
    # 对文档进行分词
    tokenized_docs = [tokenize(doc) for doc in movie_texts]
    
    # 初始化BM25
    bm25 = BM25Okapi(tokenized_docs)
    
    # 对每个查询进行检索并聚合分数
    movie_scores = defaultdict(float)
    
    for query in queries:
        # 对查询进行分词
        tokenized_query = tokenize(query)
        
        # 计算BM25分数
        scores = bm25.get_scores(tokenized_query)
        
        # 累加分数（多查询聚合）
        for idx, score in enumerate(scores):
            movie_scores[movie_ids[idx]] += score
    
    # 过滤用户已观看的电影
    user_movie_set = set(user_movie_ids)
    candidate_movies = [
        (movie_id, movie_scores[movie_id]) 
        for movie_id in movie_scores.keys() 
        if movie_id not in user_movie_set
    ]
    
    # 按分数排序
    candidate_movies.sort(key=lambda x: x[1], reverse=True)
    
    # 返回Top-K
    top_movies = candidate_movies[:top_k]
    
    # 构建推荐结果
    recommended_movies = []
    for movie_id, score in top_movies:
        movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
        recommended_movies.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'score': score
        })
    
    return recommended_movies


def recommend_movies(user_id, top_k=20, num_queries=5):
    """
    完整的推荐流程：生成查询 + BM25检索
    
    参数:
        user_id: 用户ID
        top_k: 返回前K个推荐
        num_queries: 生成的查询数量
    
    返回:
        recommended_movies: 推荐的电影列表
    """
    # 获取用户历史
    if user_id not in user_history:
        print(f"用户 {user_id} 不存在或没有历史记录")
        return []
    
    user_movie_ids = user_history[user_id]
    print(f"\n用户 {user_id} 的历史观看记录（共 {len(user_movie_ids)} 部电影）:")
    user_movies = movies[movies['movie_id'].isin(user_movie_ids[:10])]
    for _, row in user_movies.iterrows():
        print(f"  - {row['title']} ({row['genres']})")
    
    # 步骤1：生成搜索查询
    print(f"\n步骤1：正在生成搜索查询...")
    queries = generate_search_queries(user_movie_ids, num_queries=num_queries)
    
    if len(queries) == 0:
        print("无法生成查询，推荐失败")
        return []
    
    # 步骤2：使用BM25检索电影
    print(f"\n步骤2：正在使用BM25检索相关电影...")
    recommended_movies = retrieve_movies_with_bm25(queries, user_movie_ids, top_k=top_k)
    
    return recommended_movies


# 主程序
if __name__ == "__main__":
    # 示例：为指定用户推荐电影
    example_user_id = 1  # 可以修改为用户ID
    
    print("=" * 60)
    print(f"GPT4Rec 推荐系统 - 为用户 {example_user_id} 推荐电影")
    print("=" * 60)
    
    recommendations = recommend_movies(example_user_id, top_k=20, num_queries=5)
    
    if recommendations:
        print(f"\n推荐的电影（Top-{len(recommendations)}）:")
        print("-" * 60)
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie['title']} ({movie['genres']}) - 分数: {movie['score']:.4f}")
    else:
        print("\n无法生成推荐")

