from django.db import models
from openai import OpenAI

client = OpenAI(api_key="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")

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

class SEOText(models.Model):
  original_text = models.TextField()
  seo_analysis = models.TextField(null=True, blank=True)
  rewritten_text = models.TextField(null=True, blank=True)
  created_at = models.DateTimeField(auto_now_add=True)

  def __str__(self):
    return f"SEO Analysis for Text {self.id}"

class Article(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    attachment = models.FileField(upload_to='attachments/', blank=True, null=True)
    embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if self.embedding is None and self.content:
            self.embedding = generate_embedding(self.content)
        
        super().save(*args, **kwargs)
    
class Task(models.Model):
    text = models.TextField()
    embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:100]
    
    def save(self, *args, **kwargs):
        if self.embedding is None and self.text:
            self.embedding = generate_embedding(self.text)
        
        super().save(*args, **kwargs)

class GeneratedArticle(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.content[:100]
