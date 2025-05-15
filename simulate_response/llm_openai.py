# llm_openai.py
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_llm(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150):
    """
    Query OpenAI LLM with a prompt.

    Args:
        prompt: string input prompt.
        model: OpenAI model to use.
        temperature: creativity level.
        max_tokens: response length cap.

    Returns:
        response string.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"].strip()

if __name__ == "__main__":
    print(f"Key is: {os.getenv('OPENAI_API_KEY')}")