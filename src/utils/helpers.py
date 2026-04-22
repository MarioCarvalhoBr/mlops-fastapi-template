def clean_text(text: str) -> str:
    """
    Remove extra whitespace and formats the string.
    """

    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()
