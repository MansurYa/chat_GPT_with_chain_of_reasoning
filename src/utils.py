from typing import List, Dict, Tuple
import copy


def load_prompts(base_path: str = None, prompt_types: list = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Загружает полные и сокращенные версии промптов из соответствующих директорий.

    :param base_path: Базовый путь к директориям промптов (определяется автоматически, если не указан)
    :param prompt_types: Список типов промптов для загрузки (если None, загружаются все)
    :return: Кортеж из двух словарей: 1) полные промпты, 2) сокращенные промпты
    """
    import os
    import logging

    if base_path is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))

        potential_paths = [
            "",                          # Текущая директория
            "..",                        # Родительская директория
            module_dir,                  # Директория модуля
            os.path.join(module_dir, "..")  # Родительская директория модуля
        ]

        for path in potential_paths:
            prompts_dir = os.path.join(path, "prompts")
            if os.path.isdir(prompts_dir):
                base_path = path
                logging.info(f"Найдена директория промптов: {prompts_dir}")
                break
        else:
            base_path = ""
            logging.warning("Директория 'prompts' не найдена. Используется текущая директория.")

    if prompt_types is None:
        prompt_types = [
            "main_recursive_decomposition_prompt",
            "task_statement_prompt",
            "theory_gen_prompt",
            "quality_assessment_criteria_prompt",
            "start_solution_gen_prompt",
            "solution_verification_prompt",
            "action_manager_prompt",
            "final_solution_text_generator_prompt",
            "re_solve_unsuccessful_decision",
            "continue_solution_prompt",
            "decompose_task_prompt",
            "finish_task_after_solving_subtasks_prompt"
        ]

    full_prompts_dir = os.path.join(base_path, "prompts")
    shortened_prompts_dir = os.path.join(base_path, "shortened_prompts")

    full_prompts = {}
    shortened_prompts = {}

    for prompt_type in prompt_types:
        full_prompt_path = os.path.join(full_prompts_dir, f"{prompt_type}.txt")
        try:
            with open(full_prompt_path, "r", encoding="utf-8") as f:
                full_prompts[prompt_type] = f.read()
        except FileNotFoundError:
            logging.warning(f"Полный промпт не найден: {full_prompt_path}")
            full_prompts[prompt_type] = ""

        shortened_prompt_path = os.path.join(shortened_prompts_dir, f"{prompt_type}_shortened.txt")
        try:
            with open(shortened_prompt_path, "r", encoding="utf-8") as f:
                shortened_prompts[prompt_type] = f.read()
        except FileNotFoundError:
            logging.warning(f"Сокращенный промпт не найден: {shortened_prompt_path}")
            # Если сокращенная версия не найдена, используем полную
            shortened_prompts[prompt_type] = full_prompts[prompt_type]

    logging.info(f"Загружено {len(full_prompts)} полных промптов и {len(shortened_prompts)} сокращенных промптов")
    return full_prompts, shortened_prompts


def extract_text_from_content(content):
    """
    Извлекает текстовую часть из контента сообщения любого формата.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)
    return str(content)


def dedent_text(text) -> str:
    """
    Удаляет минимальное количество начальных пробелов из всех строк в многострочной строке.
    """
    # Проверка типа и преобразование к строке при необходимости
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""

    lines = text.splitlines(True)
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return ''.join([line.lstrip(' ') for line in lines])
    leading_spaces = [len(line) - len(line.lstrip(' ')) for line in non_empty_lines]
    min_spaces = min(leading_spaces)
    dedented_lines = []
    for line in lines:
        if line.strip():
            dedented_line = line[min_spaces:]
        else:
            dedented_line = line.lstrip(' ')
        dedented_lines.append(dedented_line)
    return ''.join(dedented_lines)


def add_indent(text: str, indent: int, add_vertical_bar=False) -> str:
    """
    Добавляет заданное количество пробелов в начало каждой строки в многострочной строке.
    """
    if indent < 0:
        return text
    lines = text.splitlines(True)
    if add_vertical_bar:
        lines = ['|' + line for line in lines]
    indented_lines = [' ' * indent + line for line in lines]

    return ''.join(indented_lines)


class TaskCounter:
    def __init__(self):
        self.numbers_array = []

    def increase_digit(self):
        if not self.numbers_array:
            raise Exception("Попытка увеличить номер задачи для \"Исходная\"")
        self.numbers_array[-1] += 1

    def increase_order(self):
        self.numbers_array.append(1)

    def reduce_order(self):
        if not self.numbers_array:
            raise Exception("Попытка уменьшить порядок у \"Исходная\" задача")
        self.numbers_array.pop(-1)

    def get_order(self) -> int:
        return len(self.numbers_array)

    def convert_to_str(self) -> str:
        if not self.numbers_array:
            return "Исходная"
        return "".join(f"{digit}." for digit in self.numbers_array)

    def __deepcopy__(self, memo):
        new_counter = TaskCounter()
        new_counter.numbers_array = copy.deepcopy(self.numbers_array, memo)
        return new_counter


