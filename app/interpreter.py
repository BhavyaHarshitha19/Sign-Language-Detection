"""
ASL Interpreter — converts token sequences to fluent English via GPT-4o.
Handles ASL grammar patterns and generates natural English sentences.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert American Sign Language (ASL) interpreter.

ASL has different grammar patterns from English:
- Often uses Subject-Object-Verb order
- May omit articles (a, an, the) and auxiliary verbs
- Time markers appear at beginning or end
- Spatial relationships are important

Your task:
1. Parse the ASL token sequence (may include alphabet letters and words)
2. Infer correct tense from context and time markers
3. Generate a grammatically correct, natural English sentence
4. Return ONLY the final English sentence

Examples:
  I GO STORE YESTERDAY → I went to the store yesterday.
  YOU NAME WHAT → What is your name?
  H E L L O → Hello.
  I L O V E Y O U → I love you.
  WE HAPPY TODAY → We are happy today."""


def tokens_to_sentence(tokens: list[str]) -> str:
    """Convert ASL token sequence to fluent English sentence."""
    if not tokens:
        return ""
    
    # Join tokens with spaces
    token_sequence = " ".join(tokens)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": token_sequence}
            ],
            temperature=0.2,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        # Fallback: return tokens as-is
        return " ".join(tokens).lower().capitalize() + "."
