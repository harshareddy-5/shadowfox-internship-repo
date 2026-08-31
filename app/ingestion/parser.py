import re
from typing import List, Dict, Any


class TextParser:
    """Cleans and standardizes raw extracted text while preserving section boundaries."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean unnecessary whitespace, null bytes, and non-printable control characters."""
        if not text:
            return ""

        # Remove null bytes and non-printable characters except standard whitespace
        text = text.replace("\x00", "")
        text = re.sub(r"[\r\f\v]", "\n", text)

        # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraph boundaries)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace non-newline whitespace sequences with a single space
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

        return "\n".join(lines).strip()

    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, Any]]:
        """Identify section headers in text (especially for Markdown and structured TXT)."""
        lines = text.split("\n")
        sections = []
        current_header = "Introduction / Main Body"
        current_lines = []

        for line in lines:
            # Match Markdown headers (# Header) or capitalized section titles
            header_match = re.match(r"^#{1,4}\s+(.+)$", line)
            if header_match:
                if current_lines:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_lines).strip()
                    })
                    current_lines = []
                current_header = header_match.group(1).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_lines).strip()
            })

        return [s for s in sections if s["content"]]
