from typing import Iterable

from huggingface_hub import InferenceClient, ChatCompletionOutput, ChatCompletionStreamOutput

API = "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"


def get_model_result(client: InferenceClient, messages: list[dict[str, str]]):
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=messages,
        temperature=0.2,
        max_tokens=70,
        top_p=0.9,
        stream=True
    )
    result = ""
    for chunk in stream:
        if "delta" in chunk.choices[0]:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                result += delta_content
    return result


def generate_documentation(
        current_code: str,
        similar_documentation: str | list,
        similar_code: str | list,
        show_similar_code: bool = True
) -> str:
    client = InferenceClient(api_key=API)
    similar_code = similar_code if show_similar_code else "Hidden"
    prompt = f"""Your task is to create clear and comprehensive summarization for the provided code.
    The summarization  should briefly explain what this code does. Summarization must be a connected text. Describe the functionality, not the syntax. Write a summarization for this code:
    {current_code}
    This is a similar documentation: {similar_documentation}
    Write a summarization for this code. If it is not a long code don't write a big text. Does not use introduction sentences. Don't repeat information. Start summarization from - Function verb"""
    messages = [
        {"role": "user", "content": prompt},
    ]

    result = get_model_result(client, messages)
    doc = """
    #### Summarization:
    {0}
    #### Current code:
    ```python
    {1}
    ```
    #### Similar documentation:
    {2}
    #### Similar code:
    ```python
    {3}
    ```
    """.format(result,
               current_code,
               similar_documentation[0] if isinstance(similar_documentation, list) else similar_documentation,
               similar_code[0] if isinstance(similar_code, list) else similar_code
               )
    return doc


def generate_documentation_diff_description(current_code: str, previous_code: str) -> str:
    client = InferenceClient(api_key=API)
    prompt = f"""This is previous code: 
    {previous_code}
    This is current code:
    {current_code}
    Please describe what changed in these two codes. Write in 1 sentence. Start this sentence from the verb"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    result = get_model_result(client, messages)
    diff = """
    #### Changed description
    {0}
    #### Previous code
    ```python
    {1}
    ```
    #### Current code
    ```python
    {2}
    ```
    """.format(result, previous_code, current_code)
    return diff
