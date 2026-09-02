---
title: AI Learning Roadmap Generator
emoji: 🗺️
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# AI Learning Roadmap Generator

Generates a personalized learning roadmap using Gemini, has it reviewed by
Groq, and automatically improves it based on that review — all through a
simple Gradio form.

Requires two secrets set in the Space settings (Settings -> Variables and
secrets, added as **Secrets**, not plain Variables):

- `GEMINI_API_KEY`
- `GROQ_API_KEY`
