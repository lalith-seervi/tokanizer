import re
import unicodedata


class TeluguSplitter:
    def __init__(self, corpus_path, corpus1_path):
        self.corpus = self.load_corpus(corpus_path)
        self.corpus1 = self.load_corpus(corpus1_path)
        self.limit = 2

        # Gunintham → Full vowel mapping
        self.gunintham_to_vowel = {
            "ా": "ఆ", "ి": "ఇ", "ీ": "ఈ", "ు": "ఉ", "ూ": "ఊ",
            "ె": "ఎ", "ే": "ఏ", "ై": "ఐ", "ొ": "ఒ", "ో": "ఓ", "ౌ": "ఔ",
        }

        # Telugu Unicode ranges
        self.vowels = "[అఆఇఈఉఊఋౠౡఎఏఐఒఓఔ]"
        self.consonants = "[క-హ]"
        self.chillu = "[ౘ-ౚ]"
        self.halant = "్"

        # Special chars
        self.special_chars = set(["ఽ", "ం", "ః", "౤", "౦", "౧", "౨",
                                  "౩", "౪", "౫", "౬", "౭", "౮", "౯"])

    def load_corpus(self, path):
        d = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, freq = parts
                        d[word] = int(freq)
        except FileNotFoundError:
            pass
        return d

    def normalize_word(self, w):
        w = unicodedata.normalize("NFC", w)
        return w.strip()

    def split_aksharas(self, word):
        pattern = f"({self.consonants}{self.halant}?|{self.vowels}|{self.chillu}|[ాిీుూెేైొోౌం])"
        raw = re.findall(pattern, word)
        aksharas, current = [], ""

        for piece in raw:
            if not current:
                current = piece
            else:
                if re.match(self.vowels, piece) and not current.endswith(self.halant):
                    current += piece
                else:
                    aksharas.append(current)
                    current = piece
        if current:
            aksharas.append(current)

        return aksharas

    def score_split(self, left, right):
        left_freq = self.corpus.get(left, 0) + self.corpus1.get(left, 0)
        right_freq = self.corpus.get(right, 0) + self.corpus1.get(right, 0)
        return left_freq + right_freq

    def best_split(self, word):
        aks = self.split_aksharas(word)

        if len(aks) <= 1:
            return None

        best = None
        best_score = -1

        for i in range(1, len(aks)):
            left = "".join(aks[:i])
            right = "".join(aks[i:])
            sc = self.score_split(left, right)

            if sc > best_score:
                best_score = sc
                best = (left, right, sc)

        return best

    def recursive_split(self, word):
        aks = self.split_aksharas(word)

        if len(aks) <= 3:
            return [word]

        b = self.best_split(word)
        if not b:
            return [word]

        left, right, _ = b
        return self.recursive_split(left) + self.recursive_split(right)


# ---------------------------------------------------------
#   WRAPPER THAT MAKES YOUR NOTEBOOK LOGIC FLASK-COMPATIBLE
# ---------------------------------------------------------

class TokenizerEngine:
    def __init__(self):
        self.splitter = TeluguSplitter(
            corpus_path="./vocab.txt",
            corpus1_path="./word_counts_corpus1.txt"
        )

    def process_one(self, word):
        """
        This function preserves your notebook behavior for one word.
        It returns the same data that your notebook would have printed.
        """
        word = self.splitter.normalize_word(word)
        aks = self.splitter.split_aksharas(word)

        if len(aks) <= 3:
            return {
                "word": word,
                "status": "atomic_or_short",
                "first_split": None,
                "final_parts": [word]
            }

        best = self.splitter.best_split(word)
        if not best:
            return {
                "word": word,
                "status": "no_valid_split",
                "first_split": None,
                "final_parts": [word]
            }

        left, right, score = best
        final_parts = self.splitter.recursive_split(word)

        return {
            "word": word,
            "status": "split_found",
            "first_split": {
                "left": left,
                "right": "##"+right,
                "score": score
            },
            "final_parts": final_parts
        }

    def process(self, words):
        """
        EXACT replacement of your original loop, but returns data instead of printing.
        """
        out = []
        for w in words:
            out.append(self.process_one(w))
        return out


# single global instance (used by Flask)
tokenizer = TokenizerEngine()


# ---------------------------------------------------------
#   FUNCTION CALLED BY YOUR FLASK APP
# ---------------------------------------------------------
def run_tokenizer(words: list):
    """
    The ONLY function Flask needs to call.
    Returns JSON-safe structures.
    """
    return tokenizer.process(words)


# OPTIONAL TEST (runs only when executed directly)
if __name__ == "__main__":
    sample = ["కిస్గతంలో", "మార్టినా", "ఆర్ధికశాఖ"]
    result = run_tokenizer(sample)
    import json
    print(json.dumps(result, indent=4, ensure_ascii=False))