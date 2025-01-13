def contains_korean(text: str) -> bool:
    """한글 포함 여부를 확인합니다."""
    return any('\uac00' <= char <= '\ud7a3' or '\u3131' <= char <= '\u318e' for char in text) 