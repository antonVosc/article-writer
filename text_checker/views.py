from django.http import JsonResponse
from .seo_utils import write_article
from django.shortcuts import render, get_object_or_404
from .models import SEOText, Article
import json, os
from openai import OpenAI
from dotenv import load_dotenv
from django.views.decorators.csrf import csrf_exempt
from numpy import dot
from numpy.linalg import norm
from django.views.decorators.http import require_GET, require_POST
from django.db.models import Q

def home(request):
    if request.method == "POST":
        data = json.loads(request.body)
        original_text = data.get('original_text')
        key_points = data.get('key_points', [])

        if not original_text:
            return JsonResponse({"error": "Topic is required!"}, status=400)
        
        seo_text = SEOText.objects.create(original_text=original_text)
        article = write_article(seo_text.id, original_text, key_points)

        if article:
            Article.objects.create(title=original_text, content=article)
            
            return JsonResponse({"rewritten_text": article})
        else:
            return JsonResponse({"error": "Failed to generate article."}, status=500)

    return render(request, "home.html")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@csrf_exempt
def generate_keywords(request):
    if request.method == "POST":
        data = json.loads(request.body)
        topic = data.get("topic", "")

        if not topic.strip():
            return JsonResponse({"error": "No topic provided."}, status=400)

        prompt = f"Сгенерируй 10 релевантных ключевых слов на русском языке по теме: \"{topic}\", которые наиболее SEO релеватны для этой темы в контексте родителей/детей/малышей/младенцев (только слова!)."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помогаешь с SEO и копирайтингом."},
                {"role": "user", "content": prompt}
            ]
        )

        keywords_raw = response.choices[0].message.content.strip()
        keywords = [kw.strip("•-–1234567890. ") for kw in keywords_raw.split("\n") if kw.strip()]
        
        return JsonResponse({"keywords": keywords[:10]})

    return JsonResponse({"error": "Only POST method allowed."}, status=405)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response.data[0].embedding

@require_GET
def find_related_articles(request):
    topic = request.GET.get("topic", "").strip()
    keywords = request.GET.getlist("keywords[]", [])

    query = Q(title__icontains=topic) | Q(content__icontains=topic)

    for keyword in keywords:
        query |= Q(title__icontains=keyword) | Q(content__icontains=keyword)

    articles = Article.objects.filter(query)

    data = [{
        "id": article.id,
        "title": article.title,
        "snippet": article.content[:150] + "..."
    } for article in articles[:5]]

    return JsonResponse({"articles": data})

@require_GET
def get_article(request):
    article_id = request.GET.get("id")
    article = get_object_or_404(Article, id=article_id)

    return JsonResponse({"title": article.title, "content": article.content})

def recent_articles(request):
    topic = request.GET.get("topic")

    if not topic:
        return JsonResponse({"error": "Missing topic"}, status=400)

    articles = Article.objects.filter(title__icontains=topic).order_by('-created_at')[:2]

    return JsonResponse({
        "articles": [
            {"id": a.id, "title": a.title, "content": a.content}
            for a in articles
        ]
    })

@require_POST
def delete_article(request):
    import json
    
    data = json.loads(request.body)
    article_id = data.get("id")

    if not article_id:
        return JsonResponse({"error": "Missing article ID"}, status=400)

    try:
        Article.objects.filter(id=article_id).delete()

        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)