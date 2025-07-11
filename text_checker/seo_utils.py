from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from celery import shared_task
import os, random, requests

def search_web(query, key_points, num_results=10):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        raise ValueError("Google API key or CSE ID not found in .env file.")

    full_query = f"{query} {key_points} для детей"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": full_query,
        "num": num_results,
        "hl": "ru"
    }

    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])

        if not items:
            print("No items found in Google CSE response.")
        
        return [(item["title"], item.get("snippet", ""), item["link"]) for item in items]

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")

        return []

def shorten_url(url, max_length=49):
    if len(url) <= max_length:
        return url
    
    return url[:50] + "..." + url[-19:]

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format(key_points):
    if not key_points:
        return "с выводом и советом в конце статьи"

    text = ", уделяя особое внимание следующим аспектам: "

    if len(key_points) == 1:
        return text + key_points[0]

    if len(key_points) == 2:
        return text + " и ".join(key_points)

    return text + ", ".join(key_points[:-1]) + " и " + key_points[-1]

@shared_task
def write_article(id, technical_task, key_points):
    key_points = format(key_points)
    similar_articles = retrieve_similar_articles(technical_task, top_k=5, similarity_threshold=0.92)
    context = "\n\n".join(similar_articles)
    web_sources = search_web(technical_task, key_points)
    print(web_sources)
    urls = [url for _, _, url in web_sources]
    random.shuffle(urls)

    prompt = (
        f"Напиши подробную, SEO-оптимизированную статью на тему \"{technical_task}\", ориентированную на семей с детьми, на 6000-8000 символов."
        f"Сделай акцент на слова/фразы {key_points}.\n\n"
        f"Также используй информацию: {context}.\n\n"
        f"Убедись, что статья строго соответствует требованиям всех платформ анализа текста. Вот конкретные критерии:\n\n"
        f"**Glvrd.ru:** - не менее 8 баллов\n"
        f"**Advego:** - доля вхождения стоп-слов и предлогов не более 1.5%, доля вхождения остальных слов не более 2.4%\n"
        f"**Text.ru:** - уникальность ≥ 90%, спамность ≤ 50%, водность ≤ 15%, SEO-оптимизация, читабельность, плотность ключевых слов, проверка на плагиат\n"
        f"**Turgenev (Ashmanov):** - ясность, последовательность, читаемость, SEO-принципы, ориентированные на русскоязычный текст - Общий балл не более 4\n\n"
        f"В самом конце, после заключения или последнего абзаца, добавь блок под названием 'Оценки текста:' (без кавычек, без форматирования, без дополнительных комментариев), где перечислишь оценки от всех четырех ресурсов буллетами."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are an expert technical writer specializing in creating clear, structured and SEO-optimized articles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        top_p=0.9,
    )

    rewritten_text = response.choices[0].message.content

    if urls:
        web_links_with_newlines = "\n\n".join(f"[{shorten_url(url)}]({url})" for url in urls)
        rewritten_text += "\n\nИсточники, использованные при подготовке статьи:\n\n" + web_links_with_newlines
    
    return rewritten_text

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def retrieve_similar_articles(query_text, top_k=5, similarity_threshold=0.92):
    from .models import Article

    query_embedding = generate_embedding(query_text)
    articles = Article.objects.exclude(embedding=None)
    ranked_articles = []

    for article in articles:
        sim = cosine_similarity(query_embedding, article.embedding)

        if sim > similarity_threshold:
            ranked_articles.append((sim, article))

    ranked_articles.sort(reverse=True, key=lambda x: x[0])

    return [a.content for _, a in ranked_articles[:top_k]]