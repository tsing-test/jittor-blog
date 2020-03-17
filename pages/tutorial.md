---
layout: page
title: Tutorial
description: 人越学越觉得自己无知
keywords: 教程, tutorial
comments: false
menu: 教程
permalink: /tutorial/
---

> Jittor提供以下教程，供大家学习，如有疑问，可以在Github提Issue里联系我们。

<ul class="listing">
{% for tutorial in site.tutorial %}
{% if tutorial.title != "Wiki Template" %}
<li class="listing-item"><a href="{{ site.baseurl }}{{ tutorial.url }}">{{ tutorial.title }}</a></li>
{% endif %}
{% endfor %}
</ul>
