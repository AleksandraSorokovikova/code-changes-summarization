def generate_documentation(
        current_code: str,
        similar_documentation: str | list,
        similar_code: str | list,
        show_similar_code: bool = True
) -> str:
    similar_code = similar_code if show_similar_code else "Hidden"
    doc = """
    Current code:
    {0}
    
    Similar documentation:
    {1}
    
    Similar code:
    {2}
    """.format(
        current_code,
        similar_documentation[0] if isinstance(similar_documentation, list) else similar_documentation,
        similar_code[0] if isinstance(similar_code, list) else similar_code
    )
    return doc


def generate_documentation_diff_description(current_code: str, previous_code: str) -> str:
    diff = """
    #### Previous code
    {0}
    #### Current code
    {1}
    """.format(previous_code, current_code)
    return diff