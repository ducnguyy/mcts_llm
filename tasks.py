import re
import json
import itertools
import chess
import llm_client



# Task 1: Haiku in Old English Style

# 30 topics
HAIKU_TOPICS = [
    "the morning sun",
    "autumn leaves falling",
    "a quiet stream",
    "winter snowfall",
    "cherry blossoms",
    "ocean waves",
    "a lonely mountain",
    "summer rain",
    "the full moon",
    "birds in flight",
    "a candle flame",
    "spring thunder",
    "fallen petals",
    "a frog leaping",
    "the first snowflake",
    "a spider web",
    "twilight shadows",
    "ripples on water",
    "an empty nest",
    "desert wind",
    "a bamboo forest",
    "the first firefly",
    "cicada chorus",
    "a mountain pass",
    "morning frost",
    "an old temple bell",
    "the setting sun",
    "a pine tree alone",
    "starless night",
    "autumn harvest moon",
]

_HAIKU_QUALITY_RUBRIC = (
    "You are grading a haiku written in Old English / Elizabethan style.\n"
    "Score the haiku from 0.0 to 1.0 based on THREE criteria:\n"
    "  1. Haiku structure: three lines following roughly 5, 7, 5 syllables.\n"
    "  2. Old English vocabulary: genuine use of archaic words "
    "(thee, thou, thy, hath, dost, wilt, forsooth, ere, verily, methinks, "
    "prithee, lo, behold, 'tis, etc.). A haiku with NO archaic words scores 0.\n"
    "  3. Poetic quality: imagery, coherence, and beauty of expression.\n"
    "Weight each criterion roughly equally. Be strict."
)


class HaikuTask:

    name = "haiku"

    def get_instances(self):
        return [{"topic": t, "id": i} for i, t in enumerate(HAIKU_TOPICS)]

    def get_prompt(self, instance):
        topic = instance["topic"]
        return (
            f"Write a haiku about '{topic}' using Old English words.\n"
            f"A haiku has exactly three lines:\n"
            f"  Line 1: 5 syllables\n"
            f"  Line 2: 7 syllables\n"
            f"  Line 3: 5 syllables\n"
            f"Use archaic vocabulary: thee, thou, thy, hath, dost, wilt, "
            f"forsooth, wherefore, ere, verily, lo, behold, methinks, 'tis, etc.\n"
            f"Write ONLY the three lines of the haiku. Nothing else."
        )

    def extract_answer(self, response):
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        return '\n'.join(lines[:3]) if lines else response.strip()

    def external_reward(self, instance, answer):
        return llm_client.grade_quality(_HAIKU_QUALITY_RUBRIC, answer)

    def format_result(self, instance, answer):
        return {
            "topic": instance["topic"],
            "haiku": answer,
            "external_reward": self.external_reward(instance, answer),
        }


# Task 2: Chess Move Generation
CHESS_POSITIONS = [
    {
        "id": 0,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Starting position",
    },
    {
        "id": 1,
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "description": "After 1.e4",
    },
    {
        "id": 2,
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "description": "After 1.e4 e5",
    },
    {
        "id": 3,
        "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "description": "Sicilian: after 1.e4 c5 2... Nc6",
    },
    {
        "id": 4,
        "fen": "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "description": "Alekhine's Defense: after 1.e4 Nf6",
    },
    {
        "id": 5,
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "description": "Scandinavian: after 1.e4 d5",
    },
    {
        "id": 6,
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "description": "After 1.e4 e5 2.Nf3 Nc6",
    },
    {
        "id": 7,
        "fen": "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",
        "description": "Sicilian: after 1.e4 c5 2.Nf3 Nf6",
    },
    {
        "id": 8,
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "Italian Game: after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5",
    },
    {
        "id": 9,
        "fen": "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
        "description": "Italian Game after castling",
    },
]


# Simple piece values for move quality heuristic 
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class ChessTask:

    name = "chess"

    def get_instances(self):
        return CHESS_POSITIONS

    def get_prompt(self, instance):
        fen = instance["fen"]
        desc = instance["description"]
        side = "White" if " w " in fen else "Black"
        return (
            f"Chess position (FEN): {fen}\n"
            f"Description: {desc}\n"
            f"{side} to move.\n\n"
            f"What is the best move? Respond with ONLY the move in standard "
            f"algebraic notation (e.g., e4, Nf3, Bb5, O-O, Qxd5). "
            f"Give just the move, nothing else."
        )

    def extract_answer(self, response):
        text = response.strip()
        move_pattern = re.compile(
            r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?'
            r'|O-O-O|O-O|0-0-0|0-0)\b'
        )

        matches = move_pattern.findall(text)
        if matches:
            # Return the first match
            move_str = matches[0]
            # Normalize castling notation
            move_str = move_str.replace('0-0-0', 'O-O-O').replace('0-0', 'O-O')
            return move_str

        # Fallback
        tokens = text.split()
        return tokens[0] if tokens else text

    def external_reward(self, instance, answer):
        move_str = self.extract_answer(answer)
        fen = instance["fen"]

        def _try_parse(board, s):
            try:
                m = board.parse_san(s)
                if m in board.legal_moves:
                    return m
            except Exception:
                pass
            try:
                m = chess.Move.from_uci(s)
                if m in board.legal_moves:
                    return m
            except Exception:
                pass
            return None

        try:
            board = chess.Board(fen)
            move = _try_parse(board, move_str)
            if move is None:
                return 0.0

            score = 0.3  # base for any legal move

            # Capture bonus
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured is not None:
                    score += 0.4 * (_PIECE_VALUES.get(captured.piece_type, 0) / 9.0)
                else:
                    score += 0.04  # en passant

            # Check bonus
            board.push(move)
            if board.is_check():
                score += 0.20

            # Castling bonus
            board2 = chess.Board(fen)
            if board2.is_castling(move):
                score += 0.15

            return min(1.0, score)
        except Exception:
            return 0.0

    def format_result(self, instance, answer):
        move_str = self.extract_answer(answer)
        return {
            "position": instance["description"],
            "fen": instance["fen"],
            "generated_move": move_str,
            "legal": self.external_reward(instance, answer) == 1.0,
        }

# Task 3: Game of 24

GAME24_INSTANCES = [
    {"id": 0, "numbers": [1, 2, 3, 4]}, 
    {"id": 1, "numbers": [1, 3, 5, 7]}, 
    {"id": 2, "numbers": [2, 3, 4, 5]}, 
    {"id": 3, "numbers": [1, 5, 5, 5]}, 
    {"id": 4, "numbers": [3, 3, 8, 8]}, 
    {"id": 5, "numbers": [2, 3, 5, 12]}, 
    {"id": 6, "numbers": [1, 4, 5, 6]}, 
    {"id": 7, "numbers": [4, 4, 7, 9]}, 
    {"id": 8, "numbers": [1, 2, 7, 8]}, 
    {"id": 9, "numbers": [1, 3, 4, 6]}, 
]


def _verify_expression_equals_24(expr_str, numbers):
    expr = expr_str.strip()
    if not re.match(r'^[\d\s\+\-\*/\(\)\.]+$', expr):
        return False
    used_numbers = [int(x) for x in re.findall(r'\d+', expr)]
    if sorted(used_numbers) != sorted(numbers):
        return False
    try:
        result = eval(expr)
        return abs(result - 24) < 1e-6
    except (ZeroDivisionError, SyntaxError, TypeError):
        return False


class Game24Task:

    name = "game24"

    def get_instances(self):
        return GAME24_INSTANCES

    def get_prompt(self, instance):
        nums = instance["numbers"]
        return (
            f"Use the numbers {nums[0]}, {nums[1]}, {nums[2]}, {nums[3]} "
            f"and the operations +, -, *, / to make 24.\n"
            f"You must use each number exactly once. You may use parentheses.\n"
            f"Show your reasoning, then write your final expression on the last line.\n"
            f"The final line should contain ONLY the arithmetic expression, nothing else.\n"
            f"Example format for final line: (6+2)*(4-1)"
        )

    def extract_answer(self, response):
        """Extract the arithmetic expression from the response."""
        text = response.strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return text

        for line in reversed(lines):
            line = re.sub(r'[`*]', '', line).strip()
            line = re.sub(r'^(answer|result|expression|final)\s*[:=]\s*', '', line, flags=re.IGNORECASE).strip()
            if re.match(r'^[\d\s\+\-\*/\(\)\.]+$', line) and any(c.isdigit() for c in line):
                return line

        # Fallback
        expr_pattern = re.compile(r'[\d\(\)]+[\s\+\-\*/\(\)\d]+[\d\(\)]+')
        matches = expr_pattern.findall(text)
        if matches:
            return matches[-1].strip()

        return lines[-1]

    def external_reward(self, instance, answer):
        expr = self.extract_answer(answer)
        numbers = instance["numbers"]

        if not re.match(r'^[\d\s\+\-\*/\(\.]+$', expr):
            return 0.0

        used = [int(x) for x in re.findall(r'\d+', expr)]
        if sorted(used) != sorted(numbers):
            return 0.0  # wrong numbers used

        try:
            result = eval(expr)  # safe: charset validated above
            if abs(result - 24) < 1e-6:
                return 1.0
            # Partial credit: how close to 24
            return max(0.0, 1.0 - abs(result - 24) / 24.0)
        except (ZeroDivisionError, SyntaxError, TypeError):
            return 0.0

    def format_result(self, instance, answer):
        expr = self.extract_answer(answer)
        try:
            result = eval(expr) if re.match(r'^[\d\s\+\-\*/\(\)\.]+$', expr) else None
        except Exception:
            result = None
        return {
            "numbers": instance["numbers"],
            "extracted_expression": expr,
            "evaluates_to": result,
            "correct": self.external_reward(instance, answer) == 1.0,
        }

# Task 4: GSM8K Math Word Problems
def _load_gsm8k_instances(n=30, jsonl_path=None):
    import os

    if jsonl_path is None:
        jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.jsonl")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"GSM8K test file not found at {jsonl_path}.\n"
        )

    problems = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            m = re.search(r'####\s*(-?\d[\d,]*)', item["answer"])
            if not m:
                continue
            answer = int(m.group(1).replace(',', ''))
            # Count reasoning steps
            steps = len(re.findall(r'<<', item["answer"]))
            problems.append({
                "id": i,
                "question": item["question"],
                "answer": answer,
                "steps": steps,
            })

    # Sort by difficulty
    problems.sort(key=lambda p: p["steps"])
    cutoff = int(len(problems) * 0.90)
    hard_pool = problems[len(problems) - cutoff:]   # exclude easiest
    hard_pool = hard_pool[-int(cutoff * 0.60):]     # take top 60% of remaining
    # Pick evenly spaced 30
    step = max(1, len(hard_pool) // n)
    selected = hard_pool[::step][:n]
    for i, p in enumerate(selected):
        p["id"] = i
    return selected


GSM8K_INSTANCES = _load_gsm8k_instances(n=30)


def _extract_integer_answer(response):
    text = response.strip()
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?\s*(-?\d[\d,]*)',
        r'(?:therefore|thus|so)[^.]*?(?:is|=)\s*\$?\s*(-?\d[\d,]*)',
        r'=\s*\$?\s*(-?\d[\d,]*)\s*$',
        r'\*\*(-?\d[\d,]*)\*\*\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1).replace(',', ''))
    numbers = re.findall(r'-?\d[\d,]*', text)
    if numbers:
        try:
            return int(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None


class GSM8KTask:

    name = "gsm8k"

    def get_instances(self):
        return GSM8K_INSTANCES

    def get_prompt(self, instance):
        return (
            f"Solve the following math problem step by step.\n\n"
            f"Problem: {instance['question']}\n\n"
            f"Show your reasoning clearly. On the very last line, write only:\n"
            f"The answer is: <number>"
        )

    def extract_answer(self, response):
        val = _extract_integer_answer(response)
        return str(val) if val is not None else response.strip()

    def external_reward(self, instance, answer):
        val = _extract_integer_answer(answer)
        if val is None:
            return 0.0
        return 1.0 if val == instance["answer"] else 0.0

    def format_result(self, instance, answer):
        val = _extract_integer_answer(answer)
        return {
            "question": instance["question"][:80] + "...",
            "expected": instance["answer"],
            "extracted": val,
            "correct": val == instance["answer"],
        }

# Task 5: MATH Competition Problems (from qwedsacf/competition_math)

def _extract_boxed_answer(text):
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


def _normalize_math_answer(s):
    if s is None:
        return None
    s = s.strip()
    # Remove surrounding dollar signs
    s = re.sub(r'^\$+|\$+$', '', s).strip()
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def _load_math_instances(n=30):
    from datasets import load_dataset
    ds = load_dataset('qwedsacf/competition_math', split='train')

    hard = [
        ex for ex in ds
        if ex['level'] in ('Level 4', 'Level 5')
        and _extract_boxed_answer(ex['solution']) is not None
    ]

    # Sample evenly across subject types for variety
    import random
    random.seed(42)
    random.shuffle(hard)
    selected = hard[:n]

    instances = []
    for i, ex in enumerate(selected):
        instances.append({
            'id': i,
            'problem': ex['problem'],
            'solution': ex['solution'],
            'answer': _extract_boxed_answer(ex['solution']),
            'level': ex['level'],
            'type': ex['type'],
        })
    return instances


MATH_INSTANCES = _load_math_instances(n=30)


class MATHTask:

    name = 'math'

    def get_instances(self):
        return MATH_INSTANCES

    def get_prompt(self, instance):
        return (
            f"Solve the following competition math problem.\n\n"
            f"Problem: {instance['problem']}\n\n"
            f"Show your reasoning step by step. "
            f"Put your final answer inside \\boxed{{}} on the last line, e.g.:\n"
            f"\\boxed{{42}}"
        )

    def extract_answer(self, response):
        ans = _extract_boxed_answer(response)
        return ans if ans is not None else response.strip()

    def external_reward(self, instance, answer):
        pred = _normalize_math_answer(_extract_boxed_answer(answer))
        gold = _normalize_math_answer(instance['answer'])
        if pred is None or gold is None:
            return 0.0
        return 1.0 if pred == gold else 0.0

    def format_result(self, instance, answer):
        pred = _extract_boxed_answer(answer)
        return {
            'problem': instance['problem'][:80] + '...',
            'level': instance['level'],
            'type': instance['type'],
            'expected': instance['answer'],
            'extracted': pred,
            'correct': self.external_reward(instance, answer) == 1.0,
        }


TASKS = {
    "haiku": HaikuTask(),
    "chess": ChessTask(),
    "game24": Game24Task(),
    "gsm8k": GSM8KTask(),
    "math": MATHTask(),
}

def get_task(name):
    return TASKS[name]

def get_all_tasks():
    return list(TASKS.values())