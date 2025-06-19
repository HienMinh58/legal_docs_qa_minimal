from bs4 import BeautifulSoup
import re
import html
from unicodedata import normalize

def add_space_between_number_and_char(input_string):
    # Use regular expression to match a number followed by a character and insert a space between them
    if re.search("(d3\s*k2)|(k2\s*d3)", input_string): 
        result = re.sub("(d3\s*k2)|(k2\s*d3)", "d3 k2", input_string)
    else:
        result = re.sub(r'(\d)([A-Za-z]+)(\d?)', r'\1 \2 \3', input_string)
    
    return result

def sub_multi_plus(input_string):
    result = re.sub("\++", "+", input_string)
    return result

def sub_multi_space(input_string):
    result = re.sub("\s+", " ", input_string)
    return result

def remove_special_chars(input_string):
    # Use regular expression to remove special characters except %, ., and +
    result = re.sub(r'[^\w\s%+.,]', ' ', input_string)
    result = re.sub("\s+", " ", result)
    
    return result

def remove_html_tags(text):
    if not isinstance(text, str):
        return text
    
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text(separator='\n')
    # remove \xa0
    vietnamese_lower = '0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
    stripped_text = stripped_text.replace('\xa0', ' ')
    formatted_text = re.sub(r'\n(?=[' + vietnamese_lower + '])', '', stripped_text)

    # while formatted_text.find('\n\n') != -1:
    #     formatted_text = formatted_text.replace('\n\n', '\n')
    
    return formatted_text

def decode_html_entities(text):
    decoded_text = html.unescape(text)
    decoded_text = decoded_text.replace('&nbsp;', ' ')
    decoded_text = decoded_text.replace('\n', ' ')
    
    return decoded_text


def strip_emoji(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')

    return RE_EMOJI.sub(r'', text)

def remove_special_char_at_ends(text):
    beginning_char = "–=+-,.'/\:;(•"
    end_char = ",/':.;\="
    text = text.lstrip(beginning_char)
    text = text.rstrip(end_char)

    return text

def normalize_searchterm(text):
    text = text.replace("&nbsp;", " ").replace("®", "").replace("\xa0", "").replace("\x08", "").replace("ð", "đ")

    text = normalize("NFC", text)
    return text.strip()


def normalize_text(text):
    text = text.replace(" ", " ")
    return normalize("NFC", text).strip()

def remove_html_tags(text):
    if not isinstance(text, str):
        return text
    
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text(separator='\n')
    # remove \xa0
    vietnamese_lower = '0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
    stripped_text = stripped_text.replace('\xa0', ' ')
    formatted_text = re.sub(r'\n(?=[' + vietnamese_lower + '])', '', stripped_text)

    # while formatted_text.find('\n\n') != -1:
    #     formatted_text = formatted_text.replace('\n\n', '\n')
    
    return formatted_text

def decode_html_entities(text):
    decoded_text = html.unescape(text)
    decoded_text = decoded_text.replace('&nbsp;', ' ')
    decoded_text = re.sub("\s+", " ", text)
    
    return decoded_text

def ensure_ends_with_dot(s):
    if not s.endswith('.'): 
        s += '.'       
    return s

class TextProcessor():
    def __init__(self) -> None:
        """Initialize the PreProcessor."""
        pass

    def process_searchterm(self, query):
        """Process the search term.

        Args:
            query (str): The search term to be processed.

        Returns:
            str: The processed search term.
        """
        text = normalize_searchterm(query)
        text = strip_emoji(text)
        text = remove_special_chars(text)
        # text = add_space_between_number_and_char(text)
        text = remove_special_char_at_ends(text)
        text = sub_multi_plus(text)
        text = sub_multi_space(text)
        text = text.strip().lower()
    
        return text
    
    def process_shortDescription(self, text):
        if not text:
            return ""
        text = normalize_text(text)
        text = remove_html_tags(text)
        text = decode_html_entities(text)
        text = strip_emoji(text)
        text = remove_special_chars(text)
        text = sub_multi_plus(text)
        text = sub_multi_space(text)

        return text.strip()

    def clean_text(self, text):
        if not text:
            return ""
        text = normalize_text(text)
        text = remove_html_tags(text)
        text = decode_html_entities(text)
        text = strip_emoji(text)
        text = remove_special_chars(text)
        text = sub_multi_plus(text)
        text = sub_multi_space(text)
        text = ensure_ends_with_dot(text)

        return text.strip()
    
    # def process_vaccineName(self, df):
    #     df = df.str.replace(r"[ -]", "_", regex=True)
    #     return df
    
    def process_vaccineName(self, text):
        # If the input is a single string, apply the regex replacement
        processed_text = re.sub(r"[ -]", "_", text)
        return processed_text