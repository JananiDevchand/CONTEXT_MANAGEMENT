from llm import call_llm

def summarize_markdown(markdown: str) -> str:
    prompt = f"""
Summarize the following markdown.
Preserve headings and bullet structure.
Keep it concise.

Markdown:
{markdown}
"""
    return call_llm([
        {"role": "system", "content": "You summarize documents."},
        {"role": "user", "content": prompt}
    ])
