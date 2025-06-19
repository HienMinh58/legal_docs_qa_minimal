import re

class VietnameseTextScorer:
    def __init__(self):
        self.scores = {
            'length_score': 0,
            'punctuation': 0,
            'capitalization': 0,
            'vietnamese_char_ratio': 0,
            'numeric_ratio': 0,
            'special_char_ratio': 0,
            'sentence_completeness': 0,
            "source_quality": 0,
        }

    def score(self, chunk, data_source):
        """
        Scores the quality of the Vietnamese text chunk based on multiple linguistic features.
        Returns a score between 0 (poor) and 1 (excellent).
        Args:
            chunk (str): The Vietnamese text chunk to be scored.
            data_source (str): The source of the data (e.g., 'vac', 'lc', 'chat', etc.)
        Returns:
            float: Quality score between 0 and 1
        """
        if not chunk.strip():
            return 0.0

        # Basic text statistics
        words = re.findall(r'\b\w+\b', chunk.lower())
        word_count = len(words)
        char_count = len(chunk)
        unique_words = set(words)

        # 1. Length score (normalized between 0-1, ideal 50-200 words)
        ideal_min, ideal_max = 50, 200
        if word_count <= ideal_min:
            self.scores['length_score'] = word_count / ideal_min
        elif word_count >= ideal_max:
            self.scores['length_score'] = max(0, 1 - (word_count - ideal_max) / (ideal_max * 2))
        else:
            self.scores['length_score'] = 1.0

        # 2. Punctuation score
        sentence_endings = len(re.findall(r'[.!?]+', chunk))
        if sentence_endings > 0:
            avg_words_per_sentence = word_count / sentence_endings
            # Ideal 10-25 words per sentence for Vietnamese
            if 10 <= avg_words_per_sentence <= 25:
                self.scores['punctuation'] = 1.0
            else:
                self.scores['punctuation'] = 1 - min(1, abs(avg_words_per_sentence - 17.5) / 17.5)

        # 3. Vietnamese character ratio (excluding common Latin)
        vietnamese_chars = re.findall(r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', chunk.lower())
        if char_count > 0:
            self.scores['vietnamese_char_ratio'] = min(1.0, len(vietnamese_chars) / char_count * 10)

        # 4. Numeric ratio penalty
        numbers = re.findall(r'\d+', chunk)
        if char_count > 0:
            numeric_ratio = sum(len(num) for num in numbers) / char_count
            self.scores['numeric_ratio'] = 1.0 - min(1.0, numeric_ratio * 5)  # Penalize high numeric content

        # 5. Special character ratio penalty
        special_chars = re.findall(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ.,!?]', chunk)
        if char_count > 0:
            special_char_ratio = len(special_chars) / char_count
            self.scores['special_char_ratio'] = 1.0 - min(1.0, special_char_ratio * 10)

        # 6. Sentence completeness (checks if chunk ends with sentence terminator)
        if re.search(r'[.!?][\'"]?\s*$', chunk):
            self.scores['sentence_completeness'] = 1.0
        else:
            self.scores['sentence_completeness'] = 0.3  # Partial penalty

        # 7. Score data source (FRT: [VAC, LC, TTDT], Livechat, Public)
        frt_source = ['vac', 'lc', 'ttdt']
        live_chat_source = ['chat']

        if data_source in frt_source:
            self.scores['source_quality'] = 1
        elif data_source in live_chat_source:
            self.scores['source_quality'] = 0.3
        else:
            self.scores['source_quality'] = 0.5

        # Calculate weighted final score
        weights = {
            'length_score': 0.2,
            'punctuation': 0.1,
            'vietnamese_char_ratio': 0.15,
            'numeric_ratio': 0.1,
            'special_char_ratio': 0.1,
            'sentence_completeness': 0.1,
            'source_quality': 0.25
        }

        final_score = sum(self.scores[feature] * weights[feature] for feature in weights)
        final_score = min(1.0, max(0.0, final_score))
        return float(final_score)