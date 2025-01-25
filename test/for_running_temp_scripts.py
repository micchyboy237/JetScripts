from jet.code import get_header_contents


# Example usage
if __name__ == "__main__":
    md_text = """
    # Header 1
    Content under header 1.

    ## Subheader 1.1
    Content under subheader 1.1.

    ### Subheader 1.1.1
    Content under subheader 1.1.1.

    ## Subheader 1.2
    Content under subheader 1.2.

    # Header 2
    Content under header 2.
    """

    result = get_header_contents(md_text)

    import json
    print(json.dumps(result, indent=2))
