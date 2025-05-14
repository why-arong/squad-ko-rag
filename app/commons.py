import re


def extract_answer(text):
    parts = text.split('[|assistant|]')
    if len(parts) < 2:
        return ''
    answer = parts[-1].strip()
    answer = re.split(r'\[\|.*?\|\]', answer)[0].strip()
    return answer
