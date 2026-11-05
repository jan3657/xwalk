"""Shared ChatGPT inference protocol constants."""

RESOURCE = "https://api.openai.com/v1"


class ChatGPTError(Exception):
    """A safe diagnostic that contains no credentials or authorization codes."""
