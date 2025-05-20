import unittest
import os
import tempfile
import shutil
import logging
import sys
from unittest.mock import patch, MagicMock, mock_open
from src.utils import (
    load_prompts,
    dedent_text,
    add_indent,
    TaskCounter,
    MessagesWithMetaData,
    MessageMetaData
)

from src.chat_GPT_manager_v2 import ChatLLMAgent, find_messages_by_type, _is_prompt_already_shortened, \
    _mark_prompt_as_shortened, replace_prompt_in_message, replace_prompts_by_type, safe_replace_prompt


class TestLoadPrompts(unittest.TestCase):
    """Тесты для функции load_prompts, отвечающей за загрузку промптов."""

    def setUp(self):
        """Создаем временные директории и файлы для тестирования."""
        # Создаем временную директорию
        self.temp_dir = tempfile.mkdtemp()

        # Создаем структуру директорий для промптов
        self.prompts_dir = os.path.join(self.temp_dir, "prompts")
        self.shortened_prompts_dir = os.path.join(self.temp_dir, "shortened_prompts")
        os.makedirs(self.prompts_dir)
        os.makedirs(self.shortened_prompts_dir)

        # Создаем тестовые файлы промптов
        self.test_prompts = {
            "theory_gen_prompt": "Это полная версия промпта для генерации теории. Здесь много текста...",
            "start_solution_gen_prompt": "Полная версия промпта для генерации решения. Также много текста..."
        }

        self.test_shortened_prompts = {
            "theory_gen_prompt": "Краткая версия промпта для теории.",
            "start_solution_gen_prompt": "Краткая версия промпта для решения."
        }

        # Записываем тестовые промпты в файлы
        for prompt_name, content in self.test_prompts.items():
            with open(os.path.join(self.prompts_dir, f"{prompt_name}.txt"), "w", encoding="utf-8") as f:
                f.write(content)

        for prompt_name, content in self.test_shortened_prompts.items():
            with open(os.path.join(self.shortened_prompts_dir, f"{prompt_name}_shortened.txt"), "w", encoding="utf-8") as f:
                f.write(content)

    def tearDown(self):
        """Удаляем временные файлы после тестов."""
        shutil.rmtree(self.temp_dir)

    def test_load_all_prompts(self):
        """Тест загрузки всех имеющихся промптов."""
        full_prompts, shortened_prompts = load_prompts(base_path=self.temp_dir)

        # Проверяем, что загружены правильные промпты
        for prompt_name, content in self.test_prompts.items():
            self.assertIn(prompt_name, full_prompts)
            self.assertEqual(full_prompts[prompt_name], content)

        for prompt_name, content in self.test_shortened_prompts.items():
            self.assertIn(prompt_name, shortened_prompts)
            self.assertEqual(shortened_prompts[prompt_name], content)

    def test_load_specific_prompts(self):
        """Тест загрузки только указанных промптов."""
        prompt_types = ["theory_gen_prompt"]
        full_prompts, shortened_prompts = load_prompts(base_path=self.temp_dir, prompt_types=prompt_types)

        # Проверяем, что загружен только указанный промпт
        self.assertEqual(len(full_prompts), 1)
        self.assertEqual(len(shortened_prompts), 1)
        self.assertIn("theory_gen_prompt", full_prompts)
        self.assertIn("theory_gen_prompt", shortened_prompts)

        # И что его содержимое правильное
        self.assertEqual(full_prompts["theory_gen_prompt"], self.test_prompts["theory_gen_prompt"])
        self.assertEqual(shortened_prompts["theory_gen_prompt"], self.test_shortened_prompts["theory_gen_prompt"])

    def test_missing_file_handling(self):
        """Тест обработки отсутствующих файлов."""
        # Загружаем несуществующий промпт
        prompt_types = ["nonexistent_prompt"]
        full_prompts, shortened_prompts = load_prompts(base_path=self.temp_dir, prompt_types=prompt_types)

        # Проверяем, что промпт присутствует в словарях, но с пустым содержимым
        self.assertIn("nonexistent_prompt", full_prompts)
        self.assertEqual(full_prompts["nonexistent_prompt"], "")

        # Проверяем, что для сокращенного промпта используется полная версия
        self.assertIn("nonexistent_prompt", shortened_prompts)
        self.assertEqual(shortened_prompts["nonexistent_prompt"], full_prompts["nonexistent_prompt"])

    def test_missing_shortened_file(self):
        """Тест обработки отсутствующего файла сокращенного промпта."""
        # Создаем промпт без сокращенной версии
        prompt_name = "only_full_prompt"
        with open(os.path.join(self.prompts_dir, f"{prompt_name}.txt"), "w", encoding="utf-8") as f:
            f.write("Полная версия без сокращенной.")

        prompt_types = [prompt_name]
        full_prompts, shortened_prompts = load_prompts(base_path=self.temp_dir, prompt_types=prompt_types)

        # Проверяем, что для сокращенного промпта используется полная версия
        self.assertEqual(shortened_prompts[prompt_name], full_prompts[prompt_name])


class TestMessageContextMetadata(unittest.TestCase):
    """Тесты для вспомогательных методов работы с метаданными сообщений."""

    def setUp(self):
        """Создаем тестовые метаданные и сообщения."""
        self.task_counter = TaskCounter()
        self.messages = []
        self.metadata_messages = []

        # Создаем тестовые сообщения с разными типами
        self.create_test_messages()

    def create_test_messages(self):
        """Создает тестовые сообщения с метаданными."""
        # Сообщение с типом "Theory" (обычное текстовое содержимое)
        theory_message = {
            "role": "user",
            "content": "Задача 1. [status=\"resolved\" type=\"Theory\"]\nТеоретическая база для решения..."
        }
        theory_meta = MessageMetaData(
            task_counter=self.task_counter,
            status="resolved",
            message_type="Theory",
            message=theory_message
        )
        self.messages.append(theory_message)
        self.metadata_messages.append(theory_meta)

        # Сообщение с типом "Solution" (мультимодальное содержимое)
        solution_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Задача 1. [status=\"in_progress\" type=\"Solution\"]\nРешение задачи..."
                }
            ]
        }
        self.task_counter.increase_digit()  # Переходим к задаче 2.
        solution_meta = MessageMetaData(
            task_counter=self.task_counter,
            status="in_progress",
            message_type="Solution",
            message=solution_message
        )
        self.messages.append(solution_message)
        self.metadata_messages.append(solution_meta)

        # Создаем объект MessagesWithMetaData
        self.messages_with_metadata = MessagesWithMetaData(self.messages)
        self.messages_with_metadata.metadata_messages = self.metadata_messages

    def test_find_messages_by_type(self):
        """Тест метода find_messages_by_type."""
        # Добавляем методы в MessagesWithMetaData
        setattr(MessagesWithMetaData, 'find_messages_by_type', find_messages_by_type)

        # Ищем сообщения типа "Theory"
        theory_messages = self.messages_with_metadata.find_messages_by_type("Theory")
        self.assertEqual(len(theory_messages), 1)
        self.assertEqual(theory_messages[0].type, "Theory")

        # Ищем сообщения типа "Solution"
        solution_messages = self.messages_with_metadata.find_messages_by_type("Solution")
        self.assertEqual(len(solution_messages), 1)
        self.assertEqual(solution_messages[0].type, "Solution")

        # Ищем несуществующий тип
        nonexistent_messages = self.messages_with_metadata.find_messages_by_type("NonexistentType")
        self.assertEqual(len(nonexistent_messages), 0)

    def test_is_prompt_already_shortened(self):
        """Тест метода _is_prompt_already_shortened."""
        # Добавляем метод в MessagesWithMetaData
        setattr(MessagesWithMetaData, '_is_prompt_already_shortened', _is_prompt_already_shortened)

        # Проверяем сообщение без метки shortened
        theory_meta = self.metadata_messages[0]
        self.assertFalse(self.messages_with_metadata._is_prompt_already_shortened(theory_meta))

        # Добавляем метку shortened и проверяем
        setattr(theory_meta, "shortened", True)
        self.assertTrue(self.messages_with_metadata._is_prompt_already_shortened(theory_meta))

        # Проверяем короткое сообщение (меньше 300 символов)
        solution_meta = self.metadata_messages[1]
        self.assertTrue(self.messages_with_metadata._is_prompt_already_shortened(solution_meta))

    def test_mark_prompt_as_shortened(self):
        """Тест метода _mark_prompt_as_shortened."""
        # Добавляем метод в MessagesWithMetaData
        setattr(MessagesWithMetaData, '_mark_prompt_as_shortened', _mark_prompt_as_shortened)

        # Отмечаем промпт как сокращенный
        theory_meta = self.metadata_messages[0]
        if hasattr(theory_meta, "shortened"):
            delattr(theory_meta, "shortened")  # Удаляем существующую метку, если она есть

        self.assertFalse(hasattr(theory_meta, "shortened"))
        self.messages_with_metadata._mark_prompt_as_shortened(theory_meta)
        self.assertTrue(hasattr(theory_meta, "shortened"))
        self.assertTrue(theory_meta.shortened)


class TestPromptReplacement(unittest.TestCase):
    """Тесты замены промптов и интеграции в ChatGPTAgent."""

    def setUp(self):
        """Подготовка тестовых данных."""
        # Инициализируем тестовые объекты
        self.task_counter = TaskCounter()
        self.messages = []
        self.metadata_messages = []

        # Создаем тестовые сообщения
        self.create_test_messages()

        # Патчим логгер для тестов
        self.logger_patcher = patch('logging.warning')
        self.logger_mock = self.logger_patcher.start()

    def tearDown(self):
        """Очистка после тестов."""
        self.logger_patcher.stop()

    def create_test_messages(self):
        """Создает тестовые сообщения для тестирования замены промптов."""
        # 1. Текстовое сообщение с длинным промптом
        long_theory_text = "Задача 1. [status=\"resolved\" type=\"Theory\"]\n" + "А" * 500
        theory_message = {
            "role": "user",
            "content": long_theory_text
        }
        theory_meta = MessageMetaData(
            task_counter=self.task_counter,
            status="resolved",
            message_type="Theory",
            message=theory_message
        )
        self.messages.append(theory_message)
        self.metadata_messages.append(theory_meta)

        # 2. Мультимодальное сообщение с длинным промптом
        long_solution_text = "Задача 2. [status=\"in_progress\" type=\"Solution\"]\n" + "Б" * 500
        solution_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": long_solution_text
                }
            ]
        }
        self.task_counter.increase_digit()  # Задача 2.
        solution_meta = MessageMetaData(
            task_counter=self.task_counter,
            status="in_progress",
            message_type="Solution",
            message=solution_message
        )
        self.messages.append(solution_message)
        self.metadata_messages.append(solution_meta)

        # Создаем объект MessagesWithMetaData
        self.messages_with_metadata = MessagesWithMetaData(self.messages)
        self.messages_with_metadata.metadata_messages = self.metadata_messages

    def test_replace_prompt_in_message(self):
        """Тест замены промпта в конкретном сообщении."""
        # Добавляем метод в MessagesWithMetaData
        setattr(MessagesWithMetaData, 'replace_prompt_in_message', replace_prompt_in_message)

        # Готовим сокращенный текст
        shortened_prompt = "Краткая версия теории"

        # Заменяем промпт в текстовом сообщении
        theory_meta = self.metadata_messages[0]
        result = self.messages_with_metadata.replace_prompt_in_message(theory_meta, shortened_prompt)

        # Проверяем результат
        self.assertTrue(result)
        theory_content = self.messages[0]["content"]

        # Проверяем наличие метаданных и сокращенного текста
        self.assertIn("[status=\"resolved\" type=\"Theory\"]", theory_content)
        self.assertIn(shortened_prompt, theory_content)
        self.assertNotIn("А" * 500, theory_content)  # Старый длинный текст должен исчезнуть

    def test_replace_prompts_by_type(self):
        """Тест замены всех промптов указанного типа."""
        # Добавляем необходимые методы
        setattr(MessagesWithMetaData, '_is_prompt_already_shortened', _is_prompt_already_shortened)
        setattr(MessagesWithMetaData, '_mark_prompt_as_shortened', _mark_prompt_as_shortened)
        setattr(MessagesWithMetaData, 'find_messages_by_type', find_messages_by_type)
        setattr(MessagesWithMetaData, 'replace_prompt_in_message', replace_prompt_in_message)
        setattr(MessagesWithMetaData, 'replace_prompts_by_type', replace_prompts_by_type)

        # Готовим сокращенный текст
        shortened_prompt = "Краткая версия решения"

        # Заменяем все промпты типа "Solution"
        replace_count = self.messages_with_metadata.replace_prompts_by_type("Solution", shortened_prompt)

        # Проверяем результат
        self.assertEqual(replace_count, 1)  # Должен заменить 1 промпт
        solution_content = self.messages[1]["content"][0]["text"]

        # Проверяем содержимое
        self.assertIn("[status=\"in_progress\" type=\"Solution\"]", solution_content)
        self.assertIn(shortened_prompt, solution_content)
        self.assertNotIn("Б" * 500, solution_content)  # Старый длинный текст должен исчезнуть

    def test_safe_replace_prompt(self):
        """Тест безопасной замены промптов с обработкой ошибок."""
        # Добавляем все необходимые методы
        setattr(MessagesWithMetaData, '_is_prompt_already_shortened', _is_prompt_already_shortened)
        setattr(MessagesWithMetaData, '_mark_prompt_as_shortened', _mark_prompt_as_shortened)
        setattr(MessagesWithMetaData, 'find_messages_by_type', find_messages_by_type)
        setattr(MessagesWithMetaData, 'replace_prompt_in_message', replace_prompt_in_message)
        setattr(MessagesWithMetaData, 'replace_prompts_by_type', replace_prompts_by_type)
        setattr(MessagesWithMetaData, 'safe_replace_prompt', safe_replace_prompt)

        # Тест с корректными данными
        result = self.messages_with_metadata.safe_replace_prompt("Theory", "Сокращенная теория")
        self.assertTrue(result)

        # Проверяем обработку ошибок: пустой тип сообщения
        result = self.messages_with_metadata.safe_replace_prompt("", "Сокращенный текст")
        self.assertFalse(result)

        # Проверяем обработку ошибок: пустой текст сокращенного промпта
        result = self.messages_with_metadata.safe_replace_prompt("Theory", "")
        self.assertFalse(result)

        # Проверяем обработку ошибок: несуществующий тип сообщения
        result = self.messages_with_metadata.safe_replace_prompt("NonexistentType", "Сокращенный текст")
        self.assertFalse(result)

    def test_initialize_context_optimization(self):
        """Тест инициализации методов оптимизации контекста в ChatGPTAgent."""
        # Создаем mock для ChatGPTAgent
        with patch('chat_GPT_manager_v2.ChatGPTAgent') as mock_agent:
            # Создаем имитацию MessagesWithMetaData
            mock_messages_metadata = MagicMock()
            mock_agent.messages_meta_data = mock_messages_metadata
            mock_agent.messages_meta_data.__class__ = type('MockMessagesWithMetaData', (), {})

            # Вызываем метод инициализации
            ChatLLMAgent.initialize_context_optimization(mock_agent, debug_reasoning_print=True)

            # Проверяем, что методы были добавлены
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, '_is_prompt_already_shortened'))
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, '_mark_prompt_as_shortened'))
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, 'find_messages_by_type'))
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, 'replace_prompt_in_message'))
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, 'replace_prompts_by_type'))
            self.assertTrue(hasattr(mock_agent.messages_meta_data.__class__, 'safe_replace_prompt'))


class TestIntegrationOptimization(unittest.TestCase):
    """Интеграционные тесты для оптимизации контекста."""

    @patch('chat_GPT_manager_v2.ChatGPTAgent.__call_deepseek_router_api')
    def test_optimization_in_recursive_decomposition(self, mock_api_call):
        """Тест интеграции оптимизации контекста в recursive_decomposition с минимальным использованием LLM."""
        # Настраиваем мок API для возврата предопределенных ответов
        mock_api_call.side_effect = self._mock_llm_response

        # Создаем временный каталог для промптов
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем структуру директорий
            prompts_dir = os.path.join(temp_dir, "prompts")
            shortened_dir = os.path.join(temp_dir, "shortened_prompts")
            os.makedirs(prompts_dir)
            os.makedirs(shortened_dir)

            # Создаем минимальные файлы промптов для теста
            self._create_test_prompts(prompts_dir, shortened_dir)

            # Переопределяем базовый путь для load_prompts
            with patch('utils.load_prompts', side_effect=lambda base_path=None, prompt_types=None:
                      load_prompts(base_path=temp_dir, prompt_types=prompt_types)):

                # Создаем агента
                agent = ChatLLMAgent(
                    api_key="test_key",
                    organization="test_org",
                    model_name="test_model",
                    mode=2,
                    task_prompt="Тестовый промпт",
                    max_total_tokens=10000,
                    max_response_tokens=1000,
                    temperature=0
                )

                # Тестируем инициализацию оптимизации
                agent.initialize_context_optimization()
                self.assertTrue(hasattr(agent.messages_meta_data.__class__, 'safe_replace_prompt'))

                # Запускаем рекурсивную декомпозицию с очень простой задачей
                # и минимальным количеством вызовов LLM
                with patch('chat_GPT_manager_v2.load_prompts',
                          side_effect=lambda base_path=None, prompt_types=None:
                          load_prompts(base_path=temp_dir, prompt_types=prompt_types)):

                    # Выполняем запрос с минимальной сложностью
                    test_message = "Посчитай 2+2"
                    agent.max_llm_calling_count = 3  # Ограничиваем для теста

                    # Создаем атрибут для отслеживания замен промптов
                    agent.prompts_replaced = []

                    # Патчим safe_replace_prompt для отслеживания его вызовов
                    original_safe_replace = safe_replace_prompt
                    def track_replace(self, message_type, shortened_prompt, debug_tracer=None, depth=0, strategy="all"):
                        agent.prompts_replaced.append(message_type)
                        return original_safe_replace(self, message_type, shortened_prompt, debug_tracer, depth, strategy)

                    setattr(MessagesWithMetaData, 'safe_replace_prompt', track_replace)

                    try:
                        # Запускаем декомпозицию
                        result = agent.response_from_LLM_with_hierarchical_recursive_decomposition(
                            user_message=test_message,
                            max_llm_calling_count=3,
                            debug_reasoning_print=True
                        )

                        # Проверяем были ли заменены промпты
                        self.assertGreater(len(agent.prompts_replaced), 0,
                                          "Не было выполнено ни одной замены промптов")

                        # Проверяем конкретные типы промптов для замены
                        expected_types = ["Theory", "Solution", "Quality Criteria", "Solution Verification"]
                        for expected_type in expected_types:
                            if expected_type in agent.prompts_replaced:
                                print(f"Промпт типа {expected_type} был заменен сокращенной версией")

                    except RecursionError:
                        # Ожидаемая ошибка из-за ограничения глубины
                        pass

    def _mock_llm_response(self, messages, response_format=None):
        """Создает мок-ответы LLM для тестов."""
        # Определяем, какой ответ нужен, на основе последнего сообщения
        last_message = messages[-1]

        if "Theory" in str(last_message):
            return "Теоретическая база: 2+2=4 - это базовое арифметическое действие."
        elif "Quality Criteria" in str(last_message):
            return "Критерии качества: ответ должен быть числом 4."
        elif "Solution" in str(last_message):
            return "Решение: 2+2=4"
        elif "Verification" in str(last_message):
            return "Проверка: решение верное, 2+2 действительно равно 4."
        elif "Strategy Selection" in str(last_message):
            return '{"action": "а"}'
        elif "Final Solution" in str(last_message):
            return "Ответ: 4"
        else:
            return "Стандартный ответ для теста"

    def _create_test_prompts(self, prompts_dir, shortened_dir):
        """Создает тестовые файлы промптов для интеграционного теста."""
        # Создаем основные промпты
        prompts = {
            "main_recursive_decomposition_prompt": "Основной промпт для рекурсивной декомпозиции",
            "task_statement_prompt": "Промпт для формулировки задачи",
            "theory_gen_prompt": "Промпт для теоретической базы",
            "quality_assessment_criteria_prompt": "Промпт для критериев качества",
            "start_solution_gen_prompt": "Промпт для генерации решения",
            "solution_verification_prompt": "Промпт для проверки решения",
            "action_manager_prompt": "Промпт для выбора действия",
            "final_solution_text_generator_prompt": "Промпт для финального ответа",
            "re_solve_unsuccessful_decision": "Промпт для повторного решения",
            "continue_solution_prompt": "Промпт для продолжения решения",
            "decompose_task_prompt": "Промпт для декомпозиции задачи",
            "finish_task_after_solving_subtasks_prompt": "Промпт для завершения подзадач"
        }

        # Создаем сокращенные версии
        shortened_prompts = {k: f"Сокращенная версия: {v[:20]}..." for k, v in prompts.items()}

        # Записываем файлы промптов
        for name, content in prompts.items():
            with open(os.path.join(prompts_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(content)

        # Записываем файлы сокращенных промптов
        for name, content in shortened_prompts.items():
            with open(os.path.join(shortened_dir, f"{name}_shortened.txt"), "w", encoding="utf-8") as f:
                f.write(content)


if __name__ == '__main__':
    unittest.main()
