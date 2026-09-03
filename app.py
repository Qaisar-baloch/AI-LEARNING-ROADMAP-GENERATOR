import spaces
import os
import time
import json as _json

from google import genai
from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional
import gradio as gr

# --- API keys & clients -----------------------------------------------
# On Hugging Face Spaces: Settings -> Variables and secrets -> add
# GEMINI_API_KEY and GROQ_API_KEY as Secrets (not plain Variables).

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment/secrets")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment/secrets")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Confirmed working as of your last successful Colab run.
# If this ever 404s (Google deprecates it), list available models with
# `for m in gemini_client.models.list(): print(m.name)` and swap this.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"


# --- Data models (single source of truth, defined once) ----------------

class UserProfile(BaseModel):
    domain: str
    level: str
    current_knowledge: str
    hours_per_week: float = Field(gt=0, le=40)
    goal: str
    learning_style: str
    deadline_weeks: Optional[int] = Field(default=None, gt=0)


class Topic(BaseModel):
    name: str
    description: str
    estimated_hours: float = Field(ge=0.5)


class Week(BaseModel):
    week_number: int = Field(ge=1)
    title: str
    objective: str
    topics: list[Topic]
    project: str
    total_hours: float = Field(ge=0.5)


class Roadmap(BaseModel):
    title: str
    summary: str
    total_weeks: int = Field(ge=1)
    weekly_hours: float = Field(ge=1)
    prerequisites: list[str]
    weeks: list[Week]
    final_project: str
    recommended_resources: list[str]


class RoadmapReview(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    is_acceptable: bool
    strengths: list[str]
    weaknesses: list[str]
    issues: list[str]
    recommendations: list[str]
    final_verdict: str


# --- Retry helper --------------------------------------------------------

def call_with_retry(fn, max_attempts=4, base_delay=5):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"Attempt {attempt} failed ({type(e).__name__}). Retrying in {delay}s...")
            time.sleep(delay)
    raise last_error


# --- Prompts: roadmap generation -----------------------------------------

ROADMAP_SYSTEM_PROMPT = """
You are an expert learning curriculum architect and AI education specialist.

Your job is to design realistic, structured, personalized learning roadmaps.

You must think like an experienced university curriculum designer,
professional mentor, and technical instructor.

Your roadmap must:

1. Match the learner's current skill level.
2. Respect the learner's available study time.
3. Respect the learner's deadline when provided.
4. Follow prerequisite relationships between topics.
5. Move from fundamentals to intermediate concepts and then advanced concepts.
6. Avoid teaching advanced topics before their prerequisites.
7. Include practical exercises.
8. Include a meaningful project for each major stage.
9. Include a final capstone project.
10. Avoid unnecessary topics.
11. Make the workload realistic.
12. Align the roadmap with the learner's stated career or learning goal.
13. Consider the learner's preferred learning style.
14. Clearly explain what the learner should achieve each week.

IMPORTANT:

Do not create an unrealistic "learn everything" curriculum.

Prioritize the most important skills required to achieve the learner's goal.

The learner has limited weekly study time, so prioritize depth over breadth.

Each week's total estimated hours should approximately match the learner's
available weekly study time.

The roadmap should be practical and achievable.
"""


def build_roadmap_prompt(user: UserProfile) -> str:
    deadline_text = (
        f"{user.deadline_weeks} weeks"
        if user.deadline_weeks
        else "No fixed deadline"
    )

    return f"""
Create a personalized learning roadmap for the following learner.

LEARNER PROFILE
===============

Domain:
{user.domain}

Current Skill Level:
{user.level}

Current Knowledge:
{user.current_knowledge}

Available Study Time:
{user.hours_per_week} hours per week

Learning Goal:
{user.goal}

Preferred Learning Style:
{user.learning_style}

Deadline:
{deadline_text}


ROADMAP REQUIREMENTS
====================

Design the roadmap specifically for this learner.

The roadmap should:

- Begin at the appropriate level.
- Identify important prerequisites.
- Order topics logically.
- Allocate realistic study time.
- Include theory and practical learning.
- Include projects.
- Gradually increase difficulty.
- Lead toward the learner's stated goal.
- Include a final capstone project.

Do not overload the learner.

Focus on the most important skills rather than trying to cover everything.

Generate the complete roadmap according to the required schema.
"""


def generate_roadmap(user: UserProfile) -> Roadmap:
    prompt = build_roadmap_prompt(user)

    def _call():
        return gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": ROADMAP_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": Roadmap,
            }
        )

    response = call_with_retry(_call)
    return Roadmap.model_validate_json(response.text)


# --- Roadmap rendering helpers -------------------------------------------

def roadmap_to_text(roadmap: Roadmap) -> str:
    output = []
    output.append(f"TITLE: {roadmap.title}")
    output.append(f"SUMMARY: {roadmap.summary}")
    output.append(f"TOTAL WEEKS: {roadmap.total_weeks}")
    output.append(f"HOURS PER WEEK: {roadmap.weekly_hours}")

    output.append("\nPREREQUISITES:")
    for prerequisite in roadmap.prerequisites:
        output.append(f"- {prerequisite}")

    output.append("\nWEEKLY PLAN:")
    for week in roadmap.weeks:
        output.append(f"\nWEEK {week.week_number}: {week.title}")
        output.append(f"Objective: {week.objective}")
        output.append("Topics:")
        for topic in week.topics:
            output.append(f"- {topic.name} ({topic.estimated_hours} hours): {topic.description}")
        output.append(f"Project: {week.project}")
        output.append(f"Total Hours: {week.total_hours}")

    output.append(f"\nFINAL PROJECT: {roadmap.final_project}")

    output.append("\nRECOMMENDED RESOURCES:")
    for resource in roadmap.recommended_resources:
        output.append(f"- {resource}")

    return "\n".join(output)


def roadmap_to_markdown(roadmap: Roadmap) -> str:
    lines = [f"# {roadmap.title}", "", roadmap.summary, ""]
    lines.append(f"**Total Weeks:** {roadmap.total_weeks}  ")
    lines.append(f"**Hours per Week:** {roadmap.weekly_hours}")
    lines.append("\n## Prerequisites")
    lines += [f"- {p}" for p in roadmap.prerequisites]
    lines.append("\n## Weekly Plan")
    for week in roadmap.weeks:
        lines.append(f"\n### Week {week.week_number}: {week.title}")
        lines.append(f"**Objective:** {week.objective}\n")
        lines.append("**Topics:**")
        lines += [f"- **{t.name}** ({t.estimated_hours}h) — {t.description}" for t in week.topics]
        lines.append(f"\n**Project:** {week.project}")
        lines.append(f"**Total Hours:** {week.total_hours}")
    lines.append("\n## Final Capstone Project")
    lines.append(roadmap.final_project)
    lines.append("\n## Recommended Resources")
    lines += [f"- {r}" for r in roadmap.recommended_resources]
    return "\n".join(lines)


# --- Prompts: review (Groq) ------------------------------------------

GROQ_REVIEW_SYSTEM_PROMPT = """
You are a senior AI/ML curriculum reviewer.

Your job is to critically evaluate learning roadmaps created by another
AI curriculum architect.

You must review the roadmap for:

1. Logical progression.
2. Prerequisite correctness.
3. Difficulty progression.
4. Realistic weekly workload.
5. Alignment with the learner's goal.
6. Coverage of essential skills.
7. Avoidance of unnecessary topics.
8. Quality of practical projects.
9. Quality of the final capstone project.
10. Overall career relevance.

Be critical.

Do not approve a roadmap simply because it looks good.

Identify concrete problems.

If something is missing, explain what is missing.

If something is unnecessary, explain why.

If workload is unrealistic, identify the affected weeks.

Provide actionable recommendations.

Your response must follow the required structured output format.
"""


def roadmap_to_text_condensed(roadmap: Roadmap) -> str:
    lines = [
        f"TITLE: {roadmap.title}",
        f"TOTAL WEEKS: {roadmap.total_weeks}",
        f"HOURS PER WEEK: {roadmap.weekly_hours}",
        "PREREQUISITES: " + "; ".join(roadmap.prerequisites),
        "\nWEEKS:"
    ]
    for week in roadmap.weeks:
        topic_names = ", ".join(t.name for t in week.topics)
        lines.append(
            f"Week {week.week_number}: {week.title} | Topics: {topic_names} "
            f"| Project: {week.project} | Hours: {week.total_hours}"
        )
    lines.append(f"\nFINAL PROJECT: {roadmap.final_project}")
    return "\n".join(lines)


def build_review_prompt(user: UserProfile, roadmap: Roadmap) -> str:
    roadmap_text = roadmap_to_text_condensed(roadmap)

    return f"""
Review the following AI-generated learning roadmap.

LEARNER PROFILE
===============

Domain:
{user.domain}

Level:
{user.level}

Current Knowledge:
{user.current_knowledge}

Available Hours Per Week:
{user.hours_per_week}

Goal:
{user.goal}

Learning Style:
{user.learning_style}

Deadline:
{user.deadline_weeks}


GENERATED ROADMAP
=================

{roadmap_text}
"""


def review_roadmap(user: UserProfile, roadmap: Roadmap) -> RoadmapReview:
    prompt = build_review_prompt(user, roadmap)

    def _call():
        return groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": GROQ_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "roadmap_review",
                    "schema": RoadmapReview.model_json_schema()
                }
            }
        )

    response = call_with_retry(_call)
    review_json = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        raise ValueError("Groq response was cut off before finishing the JSON.")

    data = _json.loads(review_json)
    data.setdefault("overall_score", 50)
    data.setdefault("is_acceptable", False)
    data.setdefault("strengths", [])
    data.setdefault("weaknesses", [])
    data.setdefault("issues", [])
    data.setdefault("recommendations", [])
    data.setdefault("final_verdict", "No verdict text returned by the model.")

    return RoadmapReview.model_validate(data)


# --- Prompts: improvement (Gemini) ---------------------------------------

IMPROVEMENT_SYSTEM_PROMPT = """
You are an expert AI/ML curriculum architect.

You previously generated a learning roadmap that was reviewed by a
senior curriculum reviewer.

The reviewer identified problems with the roadmap.

Your task is to improve the roadmap using the review feedback.

IMPORTANT RULES:

1. Preserve the learner's original goal.
2. Preserve the learner's available weekly study time.
3. Respect the learner's deadline.
4. Fix prerequisite problems.
5. Remove unnecessary content flagged by the reviewer.
6. Address every weakness and issue raised in the review.
7. Keep the roadmap realistic and achievable.
8. Generate the complete improved roadmap according to the required schema.
"""


def build_improvement_prompt(user: UserProfile, roadmap: Roadmap, review: RoadmapReview) -> str:
    roadmap_text = roadmap_to_text(roadmap)

    return f"""
Improve the following learning roadmap based on the review.

LEARNER PROFILE
===============

Domain:
{user.domain}

Level:
{user.level}

Current Knowledge:
{user.current_knowledge}

Available Hours Per Week:
{user.hours_per_week}

Goal:
{user.goal}

Learning Style:
{user.learning_style}

Deadline:
{user.deadline_weeks}


CURRENT ROADMAP
===============

{roadmap_text}


REVIEW FEEDBACK
===============

Overall Score: {review.overall_score}/100
Acceptable: {review.is_acceptable}

Strengths:
{chr(10).join('- ' + s for s in review.strengths)}

Weaknesses:
{chr(10).join('- ' + w for w in review.weaknesses)}

Issues:
{chr(10).join('- ' + i for i in review.issues)}

Recommendations:
{chr(10).join('- ' + r for r in review.recommendations)}

Final verdict:
{review.final_verdict}
"""


def improve_roadmap(user: UserProfile, roadmap: Roadmap, review: RoadmapReview) -> Roadmap:
    prompt = build_improvement_prompt(user, roadmap, review)

    def _call():
        return gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": IMPROVEMENT_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": Roadmap,
            }
        )

    response = call_with_retry(_call)
    return Roadmap.model_validate_json(response.text)


# --- Full pipeline ---------------------------------------------------

def run_pipeline(user: UserProfile, max_rounds: int = 1):
    print("Step 1/?: generating initial roadmap...")
    current_roadmap = generate_roadmap(user)

    for round_num in range(1, max_rounds + 1):
        print(f"Step: reviewing (round {round_num})...")
        current_review = review_roadmap(user, current_roadmap)
        if current_review.is_acceptable:
            print("Review accepted, done.")
            break
        print(f"Step: improving roadmap (round {round_num})...")
        current_roadmap = improve_roadmap(user, current_roadmap, current_review)

    return current_roadmap, current_review


# --- Gradio UI ---------------------------------------------------------
@spaces.GPU
def generate_ui(domain, level, current_knowledge, hours_per_week, goal, learning_style, deadline_weeks):
def generate_ui(domain, level, current_knowledge, hours_per_week, goal, learning_style, deadline_weeks):
    try:
        ui_user = UserProfile(
            domain=domain,
            level=level,
            current_knowledge=current_knowledge,
            hours_per_week=hours_per_week,
            goal=goal,
            learning_style=learning_style,
            deadline_weeks=int(deadline_weeks) if deadline_weeks else None
        )
        final_roadmap, final_review = run_pipeline(ui_user, max_rounds=1)
        report = roadmap_to_markdown(final_roadmap)
        report += f"\n\n---\n\n## Review\n**Score:** {final_review.overall_score}/100\n\n{final_review.final_verdict}"
        return report
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


demo = gr.Interface(
    fn=generate_ui,
    inputs=[
        gr.Textbox(label="Domain", value="Artificial Intelligence"),
        gr.Dropdown(["Beginner", "Intermediate", "Advanced"], label="Level", value="Beginner"),
        gr.Textbox(label="Current Knowledge", value="Basic Python and statistics"),
        gr.Slider(1, 40, value=10, step=1, label="Hours per week"),
        gr.Textbox(label="Goal", value="Become an AI/ML Engineer"),
        gr.Textbox(label="Learning Style", value="Project-based"),
        gr.Textbox(label="Deadline (weeks, optional)", value="24"),
    ],
    outputs=gr.Markdown(label="Roadmap"),
    title="AI Learning Roadmap Generator",
    description="Generates a personalized learning roadmap, reviews it, and improves it automatically."
)

if __name__ == "__main__":
    demo.launch()
