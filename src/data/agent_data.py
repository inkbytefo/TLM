"""
Synthetic agent training dataset generator.
Creates training examples for teaching models to use tools (Python code execution).
"""

import random
import numpy as np
from typing import List, Tuple, Iterator


# Template for agent interactions
AGENT_TEMPLATE = """SORU: {question}
DÜŞÜNCE: {thought}
EYLEM: <EXEC>{code}</EXEC>
SONUÇ: {result}
CEVAP: {answer}"""


def generate_arithmetic_problem() -> Tuple[str, str, str, str, str]:
    """
    Generate a random arithmetic problem.

    Returns:
        Tuple of (question, thought, code, result, answer)
    """
    operations = [
        ("toplama", "+", lambda a, b: a + b),
        ("çarpma", "*", lambda a, b: a * b),
        ("çıkarma", "-", lambda a, b: a - b),
        ("bölme", "//", lambda a, b: a // b if b != 0 else a),
        ("üs alma", "**", lambda a, b: a ** b if b < 10 else a ** 2),
    ]

    op_name, op_symbol, op_func = random.choice(operations)

    # Generate numbers based on operation
    if op_name == "toplama":
        a = random.randint(100, 9999)
        b = random.randint(100, 9999)
    elif op_name == "çarpma":
        a = random.randint(10, 999)
        b = random.randint(10, 999)
    elif op_name == "çıkarma":
        a = random.randint(100, 9999)
        b = random.randint(10, a)
    elif op_name == "bölme":
        b = random.randint(2, 100)
        a = b * random.randint(10, 100)
    else:  # üs alma
        a = random.randint(2, 20)
        b = random.randint(2, 6)

    result = op_func(a, b)

    question = f"{a} {op_symbol} {b} nedir?"
    thought = f"Bu bir {op_name} işlemi. Python kullanarak hesaplayacağım."
    code = f"print({a} {op_symbol} {b})"
    result_str = str(result)
    answer = f"Sonuç {result_str}'dir." if result < 1000000 else f"Sonuç {result_str}'dır."

    return question, thought, code, result_str, answer


def generate_math_function_problem() -> Tuple[str, str, str, str, str]:
    """
    Generate a mathematical function problem (factorial, fibonacci, etc.).

    Returns:
        Tuple of (question, thought, code, result, answer)
    """
    problems = [
        {
            "type": "faktöriyel",
            "question_template": "{n} faktöriyel kaçtır?",
            "thought": "Faktöriyel hesaplamak için Python math modülünü kullanacağım.",
            "code_template": "import math\nprint(math.factorial({n}))",
            "n_range": (5, 12),
        },
        {
            "type": "kare_kök",
            "question_template": "{n} sayısının karekökü nedir?",
            "thought": "Karekök hesaplamak için Python math modülünü kullanacağım.",
            "code_template": "import math\nprint(math.sqrt({n}))",
            "n_range": (100, 1000),
        },
        {
            "type": "üslü_toplam",
            "question_template": "{a}^2 + {b}^2 kaçtır?",
            "thought": "İki sayının karelerinin toplamını hesaplayacağım.",
            "code_template": "print({a}**2 + {b}**2)",
            "n_range": (10, 50),
        },
    ]

    problem = random.choice(problems)
    problem_type = problem["type"]

    if problem_type == "üslü_toplam":
        a = random.randint(*problem["n_range"])
        b = random.randint(*problem["n_range"])
        question = problem["question_template"].format(a=a, b=b)
        code = problem["code_template"].format(a=a, b=b)
        result = str(a**2 + b**2)
    else:
        n = random.randint(*problem["n_range"])
        question = problem["question_template"].format(n=n)
        code = problem["code_template"].format(n=n)

        # Calculate result
        if problem_type == "faktöriyel":
            import math
            result = str(math.factorial(n))
        else:  # kare_kök
            import math
            result = f"{math.sqrt(n):.2f}"

    thought = problem["thought"]
    answer = f"Sonuç {result}'dir." if len(result) < 10 else f"Sonuç {result}."

    return question, thought, code, result, answer


def generate_list_problem() -> Tuple[str, str, str, str, str]:
    """
    Generate a list processing problem (sum, max, min, etc.).

    Returns:
        Tuple of (question, thought, code, result, answer)
    """
    operations = [
        ("toplam", "sum", sum),
        ("en büyük", "max", max),
        ("en küçük", "min", min),
        ("ortalama", "mean", lambda lst: sum(lst) / len(lst)),
    ]

    op_name, op_func_name, op_func = random.choice(operations)

    # Generate random list
    list_size = random.randint(5, 10)
    numbers = [random.randint(1, 100) for _ in range(list_size)]
    numbers_str = str(numbers)

    result = op_func(numbers)

    if op_name == "ortalama":
        result_str = f"{result:.2f}"
    else:
        result_str = str(int(result))

    question = f"{numbers_str} listesinin {op_name}ı nedir?"
    thought = f"Liste işlemleri için Python'un built-in fonksiyonlarını kullanacağım."

    if op_name == "ortalama":
        code = f"lst = {numbers_str}\nprint(sum(lst) / len(lst))"
    else:
        code = f"lst = {numbers_str}\nprint({op_func_name}(lst))"

    answer = f"Sonuç {result_str}'dir."

    return question, thought, code, result_str, answer


def generate_agent_example() -> str:
    """
    Generate a single agent training example.

    Returns:
        Formatted training example string
    """
    # Choose random problem type
    generators = [
        generate_arithmetic_problem,
        generate_math_function_problem,
        generate_list_problem,
    ]

    generator = random.choice(generators)
    question, thought, code, result, answer = generator()

    return AGENT_TEMPLATE.format(
        question=question,
        thought=thought,
        code=code,
        result=result,
        answer=answer
    )


def get_agent_dataset(num_examples: int = 1000, seed: int = 42) -> List[str]:
    """
    Generate a dataset of agent training examples.

    Args:
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of formatted training examples
    """
    random.seed(seed)
    np.random.seed(seed)

    examples = []
    for _ in range(num_examples):
        example = generate_agent_example()
        examples.append(example)

    return examples


def save_agent_dataset(filepath: str, num_examples: int = 1000, seed: int = 42):
    """
    Generate and save agent dataset to a file.

    Args:
        filepath: Path to save the dataset
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility
    """
    examples = get_agent_dataset(num_examples, seed)

    with open(filepath, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples):
            f.write(example)
            # Add separator between examples (except last one)
            if i < len(examples) - 1:
                f.write("\n\n---\n\n")

    print(f"Saved {num_examples} agent examples to {filepath}")


def agent_dataset_iterator(num_examples: int = 1000, seed: int = 42) -> Iterator[str]:
    """
    Iterator that yields agent training examples.

    Args:
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility

    Yields:
        Formatted training example strings
    """
    random.seed(seed)
    np.random.seed(seed)

    for _ in range(num_examples):
        yield generate_agent_example()


if __name__ == "__main__":
    # Test the generator
    print("=== Testing Agent Dataset Generator ===\n")

    # Generate a few examples
    print("Example 1:")
    print(generate_agent_example())
    print("\n" + "="*50 + "\n")

    print("Example 2:")
    print(generate_agent_example())
    print("\n" + "="*50 + "\n")

    print("Example 3:")
    print(generate_agent_example())
    print("\n" + "="*50 + "\n")

    # Generate and save dataset
    print("Generating dataset...")
    save_agent_dataset("data/agent_dataset.txt", num_examples=100, seed=42)
    print("Done!")
