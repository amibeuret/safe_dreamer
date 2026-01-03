---
layout: default
---

{% for post in site.posts limit:1 %}
  <article class="post">
    <header class="post-header">
      <h1 class="post-title">{{ post.title }}</h1>
    </header>
    <div class="post-content">
      {{ post.content }}
    </div>
  </article>
{% endfor %}
