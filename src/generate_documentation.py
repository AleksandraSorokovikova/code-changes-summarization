def generate_documentation(current_code: str):
    doc = """
    {0}
    """.format(current_code)
    return doc


def generate_change_description(current_code: str, previous_code: str):
    diff = """
    #### Previous code
    {0}
    #### Current code
    {1}
    """.format(previous_code, current_code)
    return diff