import json
import re
import unicodedata

class TeluguSplitter:
    def __init__(self, corpus_path, corpus1_path):
        self.corpus = self.load_corpus(corpus_path)
        self.corpus1 = self.load_corpus(corpus1_path)
        self.limit = 2

        # Gunintham (vowel marks) → full vowels
        self.gunintham_to_vowel = {
            'ా': 'ఆ', 'ి': 'ఇ', 'ీ': 'ఈ', 'ు': 'ఉ', 'ూ': 'ఊ',
            'ె': 'ఎ', 'ే': 'ఏ', 'ై': 'ఐ', 'ొ': 'ఒ', 'ో': 'ఓ',
            'ౌ': 'ఔ', 'ం': 'అం'
        }

        # Rule 3a consonant alternations
        self.rule3a_map = {'గ': 'క', 'స': 'చ', 'డ': 'ట', 'ద': 'త', 'వ': 'ప'}

        # Sandhi reversal rules based on resulting gunintham
        self.sandhi_reversal_rules = {
            'ా': [('అ', 'అ'), ('అ', 'ఆ'), ('ఆ', 'అ'), ('ఆ', 'ఆ')],
            'ి': [('ఇ', 'ఇ')],
            'ీ': [('ఇ', 'ఇ'), ('ఇ', 'ఈ'), ('ఈ', 'ఇ'), ('ఈ', 'ఈ'), ('ఈ', 'ఆ')],
            'ు': [('ఉ', 'ఉ')],
            'ూ': [('ఉ', 'ఉ'), ('ఉ', 'ఊ'), ('ఊ', 'ఉ'), ('ఊ', 'ఊ')],
            'ె': [('అ', 'ఇ'), ('అ', 'ఈ'), ('ఇ', 'అ')],
            'ే': [('ఎ', 'ఎ'), ('ఎ', 'ఏ'), ('ఏ', 'ఎ'), ('ఏ', 'ఏ')],
            'ై': [('ఆ', 'ఇ'), ('ఆ', 'ఈ'), ('అ', 'ఏ'), ('అ', 'ఐ')],
            'ొ': [('అ', 'ఉ'), ('అ', 'ఊ'), ('ఉ', 'అ')],
            'ో': [('ఒ', 'ఒ'), ('ఒ', 'ఓ'), ('ఓ', 'ఒ'), ('ఓ', 'ఓ'), ('అః', 'అ'), ('అః', 'ఉ')],
            'ౌ': [('ఆ', 'ఉ'), ('ఆ', 'ఊ'), ('అ', 'ఓ'), ('అ', 'ఔ'), ('ఊ', 'ఆ')]
        }

        # Ottu reversal rules
        self.ottu_reversal_rules = {
            '్య': [('ఇ', 'ఉ')],  # య ottu: ఇ + ఉ = ్య
            '్వ': [('ఉ', 'అ')],  # వ ottu: ఉ + అ = ్వ
            '్ర': [('ఋ', 'అ')]   # ర ottu: ఋ + అ = ్ర
        }

    def load_corpus(self, path):
        corpus = {}
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            for i in range(0, len(lines), 2):
                corpus[lines[i]] = int(lines[i + 1])
        return corpus

    def frequency(self, word):
        return self.corpus.get(word) or self.corpus1.get(word, 0)

    def expand_ottus(self, word):
        pattern = r'([క-హ])్([క-హ])'
        def repl(m):
            return f"{m.group(1)}్+{m.group(2)}"
        return re.sub(pattern, repl, word)

    def split_aksharas(self, word):
        pattern = (
            r'([అఆఇఈఉఊఎఏఐఒఓఔఅం])'
            r'|([క-హ](?:్[క-హ])+[ాిీుూెేైొోౌ]?ం?)'
            r'|([క-హ][ాిీుూెేైొోౌ]?ం?|([క-హ]్))'
        )
        parts = re.findall(pattern, word)
        aksharas = ["".join(p) for p in parts if any(p)]
        return aksharas

    def get_gunintham(self, akshara):
        for gunintham in ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం']:
            if gunintham in akshara:
                return gunintham
        return None

    def get_gunintham_mark(self, vowel):
        vowel_to_mark = {
            'ఆ': 'ా', 'ఇ': 'ి', 'ఈ': 'ీ', 'ఉ': 'ు', 'ఊ': 'ూ',
            'ఎ': 'ె', 'ఏ': 'ే', 'ఐ': 'ై', 'ఒ': 'ొ', 'ఓ': 'ో', 'ఔ': 'ౌ'
        }
        return vowel_to_mark.get(vowel, '')

    def has_ottu(self, akshara):
        return '್' in akshara

    def get_base_consonant(self, akshara):
        base = akshara[0] if akshara else ''
        return base if 'క' <= base <= 'హ' else None

    def apply_rule1(self, word):
        if len(self.split_aksharas(word)) < self.limit:
            return [(word, None, None, 0)]
        return []

    def apply_rule2(self, word):
        if len(self.split_aksharas(word)) < self.limit:
            return []
        if word.startswith("అ"):
            rest = word[1:]
            if rest in self.corpus or rest in self.corpus1:
                f1, f2 = self.frequency("అ"), self.frequency(rest)
                score = 2 * f1 * f2 / (f1 + f2) if f1 and f2 else 0
                return [("అ", rest, f"{score:.3f}")]
        return []

    def apply_rule3(self, word):
      expanded_word = self.expand_ottus(word)
      results = []

      for i in range(1, len(expanded_word)):
          if expanded_word[i-1] == "్" or expanded_word[i] == "్":
              if expanded_word[i-1:i+1] == "్+":
                  continue
              continue

          left, right = expanded_word[:i], expanded_word[i:]

          if left.endswith("్") and not left.endswith("్+"):
              continue
          if right.startswith("్"):
              continue

          clean_left = left.replace("్+", "్").replace("+","")
          clean_right = right.replace("్+", "్").replace("+","")

          # Convert gunintham to vowel FIRST, before checking frequency
          if clean_right and clean_right[0] in self.gunintham_to_vowel:
              vowel_form = self.gunintham_to_vowel[clean_right[0]]
              final_right = vowel_form + clean_right[1:]
          else:
              final_right = clean_right
          #print(f"Possible split: '{clean_left}' + '{final_right}', f_left:{self.frequency(clean_left)},f_right {self.frequency(clean_right)}")
          # NOW check frequency and calculate score with the FINAL forms
          if (self.frequency(clean_left) > 0) and (self.frequency(final_right) > 0):
              f1, f2 = self.frequency(clean_left), self.frequency(final_right)
              score = 2 * f1 * f2 / (f1 + f2) if f1 and f2 else 0
              results.append((clean_left, final_right, f"{score:.3f}"))

      return results

    def apply_rule3a(self, word):
        results = []
        for i, ch in enumerate(word):
            if ch in self.rule3a_map:
                replaced = word[:i] + self.rule3a_map[ch] + word[i+1:]
                results.extend(self.apply_rule3(replaced))
        return results
      
    def apply_rule3b(self, word):
        results = []

        for base in ["య", "న", "ట"]:
            for mark, vowel in self.gunintham_to_vowel.items():
                pattern = base + mark
                for match in re.finditer(pattern, word):
                    start, end = match.span()
                    if start == 0 or end == len(word):
                        continue
                    replaced = word[:start] + vowel + word[end:]
                    results.extend(self.apply_rule3(replaced))
                    results.extend(self.apply_rule3a(replaced))

        if "రాలు" in word:
            replaced = word.replace("రా", "ఆ")
            results.extend(self.apply_rule3(replaced))
            results.extend(self.apply_rule3a(replaced))

        reverse_combos = {
            "ల్లా": "త్‌లా", "ల్లే": "త్‌లే", "ల్ల": "త్‌ల",
            "ఙ్మ": "క్‌మ", "శ్శ": "స్‌శ", "చ్చ": "త్‌చ",
            "జ్జ": "త్‌జ", "చ్ఛ": "త్‌శ", "క్క": "క్‌క", "ట్ట": "ట్‌ట"
        }

        for src, tgt in reverse_combos.items():
            if src in word:
                replaced = word.replace(src, tgt)
                results.extend(self.apply_rule3(replaced))
                results.extend(self.apply_rule3a(replaced))

        return results

    def apply_rule4_gunintham_splits(self, word, debug=False):
      """
      Rule 4: Split based on gunintham and ottu sandhi reversal rules.
      Set debug=True to see ALL attempted splits.
      """
      results = []
      aksharas = self.split_aksharas(word)

      if debug:
          print(f"\n{'='*60}")
          print(f"Analyzing word: {word}")
          print(f"Aksharas: {aksharas}")
          print(f"{'='*60}")

      # Build position mapping
      akshara_positions = []
      pos = 0
      for akshara in aksharas:
          akshara_positions.append((akshara, pos, pos + len(akshara)))
          pos += len(akshara)

      # Try splitting at each akshara
      for idx, (akshara, start_pos, end_pos) in enumerate(akshara_positions):
          if idx == 0:
              continue

          if debug:
              print(f"\nPosition {idx}: Akshara '{akshara}' at [{start_pos}:{end_pos}]")

          base_consonant = None
          if akshara and 'క' <= akshara[0] <= 'హ':
              base_consonant = akshara[0]

          has_ottu = '్' in akshara
          gunintham = self.get_gunintham(akshara)

          if debug:
              print(f"   Base consonant: {base_consonant if base_consonant else 'None'}")
              print(f"   Has ottu: {has_ottu}")
              print(f"   Gunintham: {gunintham if gunintham else 'None'}")

          # PART 1: Handle OTTU splits (independent from gunintham)
          if has_ottu and base_consonant:
              if debug:
                  print(f"\n   🔸 OTTU SPLITS:")

              for ottu_pattern, ottu_rules in self.ottu_reversal_rules.items():
                  if ottu_pattern in akshara:
                      if debug:
                          print(f"      Found ottu pattern: {ottu_pattern}")
                          print(f"      Rules: {ottu_rules}")

                      for v1, v2 in ottu_rules:
                          # Left: everything before + base consonant + v1 gunintham (NO ottu)
                          left_part = word[:start_pos] + base_consonant
                          v1_mark = self.get_gunintham_mark(v1)
                          if v1_mark:
                              left_part += v1_mark

                          # Right: v2 as standalone vowel + remaining
                          right_part = v2 + word[end_pos:]

                          left_freq = self.frequency(left_part)
                          right_freq = self.frequency(right_part)

                          if debug:
                              status = "Yes" if (left_freq > 0 and right_freq > 0) else "No"
                              print(f"      {status} {v1} + {v2} → {left_part} + {right_part}")
                              print(f"         Freq: L={left_freq}, R={right_freq}")

                          if left_freq > 0 and right_freq > 0:
                              score = 2 * left_freq * right_freq / (left_freq + right_freq)
                              results.append((left_part, right_part, f"{score:.3f}"))

          # PART 2: Handle GUNINTHAM splits (preserve ottu if present)
          if gunintham:
              if debug:
                  print(f"\n  GUNINTHAM SPLITS (gunintham: {gunintham}):")

              if gunintham in self.sandhi_reversal_rules:
                  sandhi_rules = self.sandhi_reversal_rules[gunintham]

                  if debug:
                      print(f"      Sandhi rules: {sandhi_rules}")

                  for v1, v2 in sandhi_rules:
                      # Construct left part
                      left_part = word[:start_pos]

                      if base_consonant:
                          # Extract the akshara without the final gunintham
                          # This preserves any ottu present
                          akshara_without_gunintham = akshara
                          if gunintham in akshara:
                              # Remove the gunintham mark
                              akshara_without_gunintham = akshara.replace(gunintham, '')

                          # Add the akshara structure (with ottu if present)
                          left_part += akshara_without_gunintham

                          # Now add v1 gunintham at the end
                          if v1 != 'అ':
                              v1_mark = self.get_gunintham_mark(v1)
                              if v1_mark:
                                  left_part += v1_mark
                          # If v1 is అ, just leave the consonant as-is (implicit అ)
                      else:
                          # Akshara is a standalone vowel
                          if v1 != 'అ':
                              left_part += v1

                      # Construct right part - prepend v2 as standalone vowel
                      remaining = word[end_pos:]
                      right_part = v2 + remaining

                      left_freq = self.frequency(left_part)
                      right_freq = self.frequency(right_part)

                      if debug:
                          status = "Yes" if (left_freq > 0 and right_freq > 0) else "No"
                          print(f"      {status} {v1} + {v2} → {left_part} + {right_part}")
                          print(f"         Freq: L={left_freq}, R={right_freq}")

                      if left_freq > 0 and right_freq > 0:
                          score = 2 * left_freq * right_freq / (left_freq + right_freq)
                          results.append((left_part, right_part, f"{score:.3f}"))

          if not has_ottu and not gunintham:
              if debug:
                  print(f"   No ottu or gunintham - skipping this akshara")

      if debug:
          print(f"\n{'='*60}")
          print(f"Total valid splits rule 4 found: {len(results)}")
          print(f"{'='*60}\n")

      return results
    def apply_rule5_filter(self, splits):
      """
      Rule 5: Filter out splits where the right/post word is just a standalone vowel.
      These are typically not meaningful morpheme boundaries.
      """
      # List of standalone vowels to reject
      standalone_vowels = {
          'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ',
          'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః'
      }

      filtered_splits = []
      rejected_count = 0

      for left, right, score in splits:
          # Check if right part is just a standalone vowel
          if right in standalone_vowels:
              rejected_count += 1
              continue  # Skip this split

          filtered_splits.append((left, right, score))

      return filtered_splits, rejected_count


    def test_word_detailed(self, word, debug=True):
      """
      Test a word and show detailed frequency breakdown for each split.
      """
      print(f"\n{'='*80}")
      print(f"Testing word: {word}")
      print(f"Aksharas: {self.split_aksharas(word)}")
      print('='*80)

      splits = []
      splits.extend(self.apply_rule2(word))
      splits.extend(self.apply_rule3(word))
      splits.extend(self.apply_rule3a(word))
      splits.extend(self.apply_rule3b(word))
      splits.extend(self.apply_rule4_gunintham_splits(word, debug=debug))
      if splits:
        splits, rejected = self.apply_rule5_filter(splits)
        if rejected > 0:
            print(f"\n Rule 5: Filtered out {rejected} split(s) with standalone vowel as post-word")

      if not splits:
          print("No splits found")
          return

      print(f"\nFound {len(splits)} splits:\n")
      print(f"{'Left Part':<20} {'Right Part':<20} {'Left Freq':<12} {'Right Freq':<12} {'HM Score':<12}")
      print("-" * 88)

      for left, right, score in splits:
          left_freq = self.frequency(left)
          right_freq = self.frequency(right)

          # Extract numeric score
          if isinstance(score, str) and ":" in score:
              numeric_score = float(score.split(":")[-1])
          else:
              numeric_score = float(score)

          # Calculate harmonic mean manually for verification
          if left_freq > 0 and right_freq > 0:
              hm = 2 * left_freq * right_freq / (left_freq + right_freq)
          else:
              hm = 0

          print(f"{left:<20} {right:<20} {left_freq:<12} {right_freq:<12} {numeric_score:<12.3f}")
      return splits

    def process_words(self, limit=10, debug_rule4=False):
        words = list(self.corpus.keys())[:limit]
        results = []

        for word in words:
            splits = []

            r1 = self.apply_rule1(word)
            if r1:
                results.append((word, "No split (aksharas<3)", r1))
                continue

            splits.extend(self.apply_rule2(word))
            splits.extend(self.apply_rule3(word))
            splits.extend(self.apply_rule3a(word))
            splits.extend(self.apply_rule3b(word))
            splits.extend(self.apply_rule4_gunintham_splits(word, debug=debug_rule4))
            if splits:
              splits, rejected = self.apply_rule5_filter(splits)
              if rejected > 0:
                  print(f"\n Rule 5: Filtered out {rejected} split(s) with standalone vowel as post-word")

            if splits:
                results.append((word, "Valid split", splits))
            else:
                results.append((word, "No valid split", []))

        return results

def extract_score(s):
    try:
        if isinstance(s, str) and ":" in s:
            return float(s.split(":")[-1])
        return float(s)
    except:
        return 0.0

def akshara_count(s):
    global splitter
    return len(splitter.split_aksharas(s))


def find_best_split_for(word, first_level=True):
    global splitter
    orig_limit = getattr(splitter, "limit", 2)
    splitter.limit = 2 if first_level else 4

    cand = []
    cand.extend(splitter.apply_rule2(word))
    cand.extend(splitter.apply_rule3(word))
    cand.extend(splitter.apply_rule3a(word))
    cand.extend(splitter.apply_rule3b(word))
    cand.extend(splitter.apply_rule4_gunintham_splits(word, debug=False))

    if cand:
        cand, _ = splitter.apply_rule5_filter(cand)

    splitter.limit = orig_limit

    if not cand:
        return None

    return max(cand, key=lambda x: extract_score(x[2]))


def recursive_split_part(part):
    global splitter
    if akshara_count(part) < 4:
        return [part]

    bs = find_best_split_for(part, first_level=False)
    if not bs:
        return [part]

    L, R, _ = bs
    return recursive_split_part(L) + recursive_split_part(R)


def get_split_for_word(word, corpus_path="./vocab.txt", corpus1_path="./word_counts_corpus1.txt"):
    """
    Process a single word and return JSON structure.
    """
    global splitter
    splitter = TeluguSplitter(corpus_path, corpus1_path)

    # Too small
    if len(splitter.split_aksharas(word)) < 3:
        return {
            "word": word,
            "status": "Too small",
            "best_split": None,
            "recursive_parts": [word]
        }

    splits = []
    splits.extend(splitter.apply_rule2(word))
    splits.extend(splitter.apply_rule3(word))
    splits.extend(splitter.apply_rule3a(word))
    splits.extend(splitter.apply_rule3b(word))
    splits.extend(splitter.apply_rule4_gunintham_splits(word, debug=False))

    if splits:
        splits, _ = splitter.apply_rule5_filter(splits)

    if not splits:
        return {
            "word": word,
            "status": "No valid split",
            "best_split": None,
            "recursive_parts": [word]
        }

    best = max(splits, key=lambda x: extract_score(x[2]))
    L, R, score = best

    final_parts = recursive_split_part(L) + recursive_split_part(R)

    return {
        "word": word,
        "status": "Valid split",
        "best_split": {
            "left": L,
            "right": R,
            "score": float(score)
        },
        "recursive_parts": final_parts
    }


def run_tokenizer(words_list, corpus_path="./vocab.txt", corpus1_path="./word_counts_corpus1.txt"):
    """
    Input:  ["అనుమానించడం", "పరీక్ష"]
    Output: [ {...}, {...} ]
    """
    global splitter
    splitter = TeluguSplitter(corpus_path, corpus1_path)

    results = []
    for word in words_list:
        # process word using existing logic
        result = get_split_for_word(word, corpus_path, corpus1_path)
        results.append(result)
    return results
