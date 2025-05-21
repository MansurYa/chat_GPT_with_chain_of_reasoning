import json
import traceback
from datetime import datetime
from rich.console import Console
import textwrap
from typing import Optional, Dict, Any, List, Union, Tuple
import tempfile
import os

from src.utils import TaskCounter
from src.messages_meta_data_manager import MessagesWithMetaData, MessageMetaData


class DebugTracer:
    """
    Трассировщик для отладки рекурсивных алгоритмов с LLM.
    Обеспечивает двойной вывод: форматированное отображение в консоли и
    надежное сохранение в JSONL-файл для последующего анализа.
    """

    def __init__(
        self,
        log_folder: str = "../logs",
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,
        console_preview_length: int = 75,
        messages_meta_data: Optional[MessagesWithMetaData] = None,
        debug_numbering: bool = False,
    ):
        """
        Инициализирует трассировщик для отладки рекурсивных алгоритмов.

        :param log_folder: Директория для хранения лог-файлов. Путь относительно расположения
                          скрипта или абсолютный путь.
        :param enable_console: Включить форматированный вывод в консоль.
        :param max_file_size: Максимальный размер файла до ротации (в байтах).
        :param console_preview_length: Длина превью текста в консоли.
        :param messages_meta_data: Объект MessagesWithMetaData для отслеживания иерархии задач.
        :param debug_numbering: Включить подробное логирование процесса нумерации задач.
        :raises IOError: Если невозможно создать директорию для логов.
        """
        try:
            abs_log_folder = os.path.abspath(log_folder)
            os.makedirs(abs_log_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file_base = os.path.join(abs_log_folder, f"recursion_{timestamp}")
            self.log_file = f"{self.log_file_base}.jsonl"

        except (IOError, PermissionError) as e:
            temp_dir = tempfile.gettempdir()
            print(f"Ошибка при создании директории {log_folder}: {e}. Используется временная директория: {temp_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file_base = os.path.join(temp_dir, f"recursion_{timestamp}")
            self.log_file = f"{self.log_file_base}.jsonl"

        self.file_counter = 0
        self.max_file_size = max_file_size
        self.console_preview_length = console_preview_length
        self.messages_meta_data = messages_meta_data
        self.debug_numbering = debug_numbering

        self.console = Console() if enable_console else None
        self.msg_counter = 0

        self.depth_counters = {}
        self.phase_styles = {
            "Instruction": ("📋", "bright_white"),
            "Task Statement": ("📝", "cyan"),
            "Theory": ("📚", "blue"),
            "Quality Criteria": ("🎯", "bright_blue"),
            "Solution": ("🔍", "green"),
            "Solution Verification": ("✅", "yellow"),
            "Strategy Selection": ("🧠", "magenta"),
            "Task Decomposition": ("🧩", "orange3"),
            "Solution Retry": ("🔄", "steel_blue"),
            "Solution Continuation": ("➡️", "dodger_blue"),
            "Final Solution": ("🏆", "gold1"),
            "Error": ("⚠️", "red"),
            "Context": ("📚", "purple"),
            "Trim Context": ("✂️", "yellow"),
        }

        # Сохраняем последний использованный task_counter для поддержания согласованности
        # при переходе между фазами одной задачи
        self.last_used_task_counter = None
        self.last_hierarchy_id = None
        self.phase_to_hierarchy_map = {}

        if self.console:
            self.console.print(f"[bold green]Трассировщик инициализирован[/] 📊 [{timestamp}]")
            self.console.print(f"Логи сохраняются в: [italic]{self.log_file}[/]")
            if self.messages_meta_data:
                self.console.print(f"[bold cyan]Интегрирован с MessagesWithMetaData[/]")
            if self.debug_numbering:
                self.console.print(f"[bold magenta]Режим отладки нумерации включен[/]")

    def set_messages_meta_data(self, messages_meta_data: MessagesWithMetaData) -> None:
        """
        Устанавливает или обновляет объект MessagesWithMetaData.

        :param messages_meta_data: Объект MessagesWithMetaData для отслеживания иерархии задач.
        """
        self.messages_meta_data = messages_meta_data
        if self.console:
            self.console.print(f"[bold cyan]MessagesWithMetaData обновлен[/]")
            if hasattr(messages_meta_data, 'task_counter'):
                self.console.print(f"[cyan]Текущий TaskCounter: {messages_meta_data.task_counter.convert_to_str()}[/]")

    def get_current_task_counter(self) -> Optional[TaskCounter]:
        """
        Получает текущий TaskCounter из MessagesWithMetaData, если доступен.

        :return: Объект TaskCounter или None, если недоступен.
        """
        if self.messages_meta_data is None:
            if self.debug_numbering:
                self.console.print("[yellow]get_current_task_counter: messages_meta_data is None[/]")
            return None

        try:
            # Возвращаем непосредственно текущий task_counter (без создания копии)
            # для обеспечения согласованности при обновлениях
            if hasattr(self.messages_meta_data, 'task_counter'):
                task_counter = self.messages_meta_data.task_counter
                if self.debug_numbering:
                    self.console.print(f"[dim green]get_current_task_counter: {task_counter.convert_to_str()}[/]")
                return task_counter
            else:
                if self.debug_numbering:
                    self.console.print("[yellow]get_current_task_counter: task_counter not found in messages_meta_data[/]")
                return None

        except (AttributeError, IndexError) as e:
            if self.debug_numbering:
                self.console.print(f"[red]get_current_task_counter error: {str(e)}[/]")
            return None

    def find_meta_for_phase(self, phase: str) -> Optional[MessageMetaData]:
        """
        Ищет MessageMetaData, соответствующий указанной фазе.

        Метод ищет самый последний экземпляр сообщения указанного типа
        в metadata_messages, что позволяет корректно определить иерархию
        для текущей фазы алгоритма.

        :param phase: Фаза/этап алгоритма для поиска в метаданных.
        :return: Объект MessageMetaData или None, если не найден.
        """
        if self.messages_meta_data is None or not hasattr(self.messages_meta_data, 'metadata_messages'):
            if self.debug_numbering:
                self.console.print(f"[yellow]find_meta_for_phase({phase}): metadata_messages недоступны[/]")
            return None

        try:
            # Расширенная карта соответствия фаз и типов сообщений
            phase_to_type_mapping = {
                "Task Statement": "Task Statement",
                "Theory": "Theory",
                "Quality Criteria": "Quality Criteria",
                "Solution": "Solution",
                "Solution Verification": "Solution Verification",
                "Strategy Selection": "Strategy Selection",
                "Task Decomposition": "Task Decomposition",
                "Solution Retry": "Solution Retry",
                "Solution Continuation": "Solution Continuation",
                "Final Solution": "Final Solution",
                "Instruction": "Instruction",
                "Error": "Error",
                "Task Integration": "Task Integration",
                "Subtask Start": "Subtask Start",
                "Subtask Complete": "Subtask Complete",
                "Default": None,  # Для фаз без соответствующего типа
            }

            message_type = phase_to_type_mapping.get(phase, phase)

            # Ищем последнее сообщение с соответствующим типом
            found_meta = None
            for i, meta in enumerate(reversed(self.messages_meta_data.metadata_messages)):
                meta_type = getattr(meta, 'type', None)
                if meta_type == message_type:
                    found_meta = meta
                    if self.debug_numbering:
                        status = getattr(meta, 'status', 'unknown')
                        hierarchy = getattr(meta.task_number, 'convert_to_str', lambda: "unknown")()
                        self.console.print(f"[dim cyan]find_meta_for_phase({phase}): Найдено [{hierarchy}] {meta_type} [{status}][/]")
                    break

            # Если ничего не найдено, но phase есть в нашей карте фаза-иерархия,
            # используем последний известный task_counter для этой фазы
            if found_meta is None and phase in self.phase_to_hierarchy_map:
                hierarchy_info = self.phase_to_hierarchy_map[phase]
                if self.debug_numbering:
                    self.console.print(f"[yellow]find_meta_for_phase({phase}): Используется сохраненная иерархия: {hierarchy_info}[/]")
                return None  # Мы не можем создать MessageMetaData, но сохраненная иерархия будет использована

            return found_meta
        except Exception as e:
            if self.debug_numbering:
                self.console.print(f"[red]find_meta_for_phase error: {str(e)}[/]")
                print(traceback.format_exc())
            return None

    def log(
        self,
        depth: int,
        phase: str,
        prompt: str,
        response: str | None = None,
        extra: dict | None = None,
        message_meta: Optional[MessageMetaData] = None,
    ) -> None:
        """
        Логирует взаимодействие с LLM.

        :param depth: Глубина рекурсии.
        :param phase: Фаза/этап алгоритма.
        :param prompt: Промпт, отправленный к LLM.
        :param response: Ответ от LLM (None, если запрос не выполнялся).
        :param extra: Дополнительные метаданные (время выполнения, коды действий и т.д.).
        :param message_meta: Объект MessageMetaData для логирования.
        """
        self._check_file_rotation()

        # Основная логика определения иерархии задачи
        task_counter, hierarchy_id, meta_status, meta_type = self._determine_hierarchy_for_log(depth, phase, message_meta)

        # Формируем базовую запись лога
        entry = {
            "ts": datetime.now().isoformat(),
            "log_type": "interaction",
            "depth": depth,
            "phase": phase,
            "hierarchy": hierarchy_id,
            "msg_id": self.msg_counter,
            "prompt": prompt,
            "prompt_preview": prompt[:self.console_preview_length]
            + ("..." if len(prompt) > self.console_preview_length else ""),
            "prompt_length": len(prompt),
        }

        if meta_status is not None:
            entry["meta_status"] = meta_status
        if meta_type is not None:
            entry["meta_type"] = meta_type

        if task_counter is not None:
            entry["task_counter"] = task_counter.convert_to_str()
            entry["task_order"] = task_counter.get_order()
            if hasattr(task_counter, 'numbers_array'):
                entry["task_numbers"] = task_counter.numbers_array

        try:
            entry["prompt_tokens"] = len(prompt.split())
            if response:
                entry["response_tokens"] = len(response.split())
        except Exception:
            pass

        if response:
            entry["response"] = response
            entry["response_preview"] = response[:self.console_preview_length] + (
                "..." if len(response) > self.console_preview_length else ""
            )
            entry["response_length"] = len(response)

        if extra:
            entry.update(extra)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Ошибка записи лога: {e}[/]")
            self.log_error(depth, f"Ошибка записи лога: {str(e)}")

        if self.console:
            self._print_to_console(entry)

        self.msg_counter += 1

    def _determine_hierarchy_for_log(
        self, depth: int, phase: str, message_meta: Optional[MessageMetaData] = None
    ) -> Tuple[Optional[TaskCounter], str, Optional[str], Optional[str]]:
        """
        Определяет иерархию задачи и соответствующие метаданные для записи лога.

        Эта функция содержит основную логику определения иерархии, вынесенную из
        метода log для лучшей организации кода.

        :param depth: Глубина рекурсии.
        :param phase: Фаза/этап алгоритма.
        :param message_meta: Объект MessageMetaData для логирования.
        :return: Кортеж (task_counter, hierarchy_id, meta_status, meta_type)
        """
        task_counter = None
        meta_status = None
        meta_type = None

        # Отладка начала определения иерархии
        if self.debug_numbering:
            self.console.print(f"[dim]_determine_hierarchy_for_log: phase={phase}, depth={depth}[/]")

        # 1. Если message_meta явно предоставлен, используем его
        if message_meta is not None:
            task_counter = getattr(message_meta, 'task_number', None)
            meta_status = getattr(message_meta, 'status', None)
            meta_type = getattr(message_meta, 'type', None)

            if self.debug_numbering and task_counter:
                self.console.print(f"[dim green]Используется явно предоставленный message_meta: {task_counter.convert_to_str()}[/]")

        # 2. Если message_meta не предоставлен, пытаемся найти его по фазе
        elif self.messages_meta_data is not None:
            found_meta = self.find_meta_for_phase(phase)

            if found_meta is not None:
                task_counter = getattr(found_meta, 'task_number', None)
                meta_status = getattr(found_meta, 'status', None)
                meta_type = getattr(found_meta, 'type', None)

                if self.debug_numbering and task_counter:
                    self.console.print(f"[dim green]Найден соответствующий MessageMetaData: {task_counter.convert_to_str()}[/]")

            # 3. Если не найден MessageMetaData, используем текущий TaskCounter
            elif hasattr(self.messages_meta_data, 'task_counter'):
                task_counter = self.messages_meta_data.task_counter

                if self.debug_numbering:
                    self.console.print(f"[dim yellow]Используется текущий TaskCounter: {task_counter.convert_to_str()}[/]")

        # 4. Проверка наличия сохраненной иерархии для фазы
        if task_counter is None and phase in self.phase_to_hierarchy_map:
            # Для согласованности используем последнюю известную иерархию для этой фазы
            hierarchy_id = self.phase_to_hierarchy_map[phase]

            if self.debug_numbering:
                self.console.print(f"[dim yellow]Используется сохраненная иерархия для фазы {phase}: {hierarchy_id}[/]")

            if self.last_used_task_counter is not None:
                task_counter = self.last_used_task_counter

        # 5. Если task_counter все еще None, и у нас есть last_used_task_counter
        if task_counter is None and self.last_used_task_counter is not None:
            task_counter = self.last_used_task_counter

            if self.debug_numbering:
                self.console.print(f"[dim yellow]Используется последний известный TaskCounter: {task_counter.convert_to_str()}[/]")

        # 6. Наконец, получаем hierarchy_id из task_counter или fallback
        hierarchy_id = self._get_hierarchy_id(depth, task_counter)

        # 7. Сохраняем последний использованный counter и иерархию
        if task_counter is not None:
            self.last_used_task_counter = task_counter

        self.last_hierarchy_id = hierarchy_id

        # 8. Сохраняем соответствие фазы и иерархии для последующего использования
        self.phase_to_hierarchy_map[phase] = hierarchy_id

        return task_counter, hierarchy_id, meta_status, meta_type

    def log_error(self, depth: int, error_msg: str, context: str | None = None, message_meta: Optional[MessageMetaData] = None) -> None:
        """
        Логирует ошибку в алгоритме.

        :param depth: Глубина рекурсии.
        :param error_msg: Сообщение об ошибке.
        :param context: Контекст ошибки (например, стек вызовов).
        :param message_meta: Объект MessageMetaData для логирования.
        """
        extra = {"error": True, "log_type": "error"}
        if context:
            extra["context"] = context

        self.log(depth=depth, phase="Error", prompt=error_msg, extra=extra, message_meta=message_meta)

    def log_messages_context(self, messages_meta_data: Optional[MessagesWithMetaData] = None) -> None:
        """
        Сохраняет весь контекст модели (все сообщения с метаданными) в лог-файл.

        :param messages_meta_data: Объект MessagesWithMetaData для логирования.
                                 Если None, используется self.messages_meta_data.
        """
        # Используем переданный объект или внутренний
        meta_data = messages_meta_data or self.messages_meta_data

        if not meta_data or not hasattr(meta_data, 'metadata_messages'):
            if self.console:
                self.console.print("[yellow]Не удалось логировать контекст: отсутствуют метаданные сообщений[/]")
            return

        self._check_file_rotation()

        # Логируем заголовок контекста
        try:
            header_entry = {
                "ts": datetime.now().isoformat(),
                "log_type": "context_header",
                "description": "Начало логирования контекста модели",
                "messages_count": len(meta_data.metadata_messages),
                "task_counter": meta_data.task_counter.convert_to_str() if hasattr(meta_data, 'task_counter') else "unknown"
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(header_entry, ensure_ascii=False) + "\n")
                f.flush()

            if self.console:
                self.console.print(f"[bold purple]Логирование контекста: {len(meta_data.metadata_messages)} сообщений[/]")
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Ошибка при логировании заголовка контекста: {e}[/]")
            return

        # Логируем каждое сообщение в контексте
        for i, meta_msg in enumerate(meta_data.metadata_messages):
            try:
                # Извлекаем информацию из MessageMetaData
                task_number = meta_msg.task_number
                status = getattr(meta_msg, 'status', 'unknown')
                msg_type = getattr(meta_msg, 'type', 'unknown')
                message = getattr(meta_msg, 'message', {})

                role = message.get('role', 'unknown')
                content = self._extract_content(message.get('content', ''))

                message_entry = {
                    "ts": datetime.now().isoformat(),
                    "log_type": "context_message",
                    "index": i,
                    "role": role,
                    "hierarchy": task_number.convert_to_str() if task_number else "unknown",
                    "task_order": task_number.get_order() if task_number else -1,
                    "task_numbers": task_number.numbers_array if task_number and hasattr(task_number, 'numbers_array') else [],
                    "status": status,
                    "type": msg_type,
                    "content": content,
                    "content_preview": content[:self.console_preview_length] + ("..." if len(content) > self.console_preview_length else ""),
                    "content_length": len(content)
                }

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(message_entry, ensure_ascii=False) + "\n")
                    f.flush()

                # Опционально выводим в консоль (краткую информацию)
                if self.console:
                    # Ограничиваем вывод контекста, чтобы не перегружать консоль
                    if i % 5 == 0 or i == len(meta_data.metadata_messages) - 1:
                        self.console.print(f"[dim]Сообщение {i+1}/{len(meta_data.metadata_messages)}: {role}/{msg_type} [{status}][/]")

            except Exception as e:
                error_info = traceback.format_exc()
                if self.console:
                    self.console.print(f"[bold red]Ошибка при логировании сообщения {i}: {e}[/]")

                error_entry = {
                    "ts": datetime.now().isoformat(),
                    "log_type": "context_error",
                    "index": i,
                    "error": str(e),
                    "traceback": error_info
                }

                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                        f.flush()
                except:
                    pass

                continue

        try:
            footer_entry = {
                "ts": datetime.now().isoformat(),
                "log_type": "context_footer",
                "description": "Конец логирования контекста модели",
                "processed_messages": len(meta_data.metadata_messages)
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(footer_entry, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Ошибка при логировании окончания контекста: {e}[/]")

    def log_trimmed_messages(self, original_messages: list, trimmed_messages: list) -> None:
        """
        Логирует информацию о сообщениях после обрезки контекста.

        :param original_messages: Исходный список сообщений.
        :param trimmed_messages: Обрезанный список сообщений.
        """
        self._check_file_rotation()

        try:
            # Проверяем валидность входных данных
            if not isinstance(original_messages, list) or not isinstance(trimmed_messages, list):
                if self.console:
                    self.console.print("[bold red]Ошибка: параметры должны быть списками[/]")
                return

            # Определяем количество удаленных сообщений
            removed_count = len(original_messages) - len(trimmed_messages)

            # Если ничего не было удалено, просто выходим
            if removed_count <= 0:
                if self.console:
                    self.console.print("[dim]Обрезка контекста: сообщения не были удалены[/]")
                return

            # Основная информация об обрезке
            entry = {
                "ts": datetime.now().isoformat(),
                "log_type": "trimmed_context",
                "phase": "Trim Context",
                "original_messages_count": len(original_messages),
                "trimmed_messages_count": len(trimmed_messages),
                "removed_messages_count": removed_count,
            }

            # Определяем тип первого сообщения (system или user)
            start_index = 1 if original_messages and len(original_messages) > 0 and original_messages[0].get("role") == "system" else 0

            # Список удаленных сообщений (только роли и предпросмотр содержимого)
            removed_messages = []
            for i in range(start_index, len(original_messages) - len(trimmed_messages) + start_index):
                if i >= len(original_messages):
                    break  # Защита от выхода за границы списка

                msg = original_messages[i]
                content = self._extract_content(msg.get("content", ""))

                # Сохраняем метаданные о сообщении
                msg_data = {
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
                }

                # Добавляем связанные метаданные, если доступны
                if self.messages_meta_data and hasattr(self.messages_meta_data, 'metadata_messages'):
                    if i < len(self.messages_meta_data.metadata_messages):
                        meta_msg = self.messages_meta_data.metadata_messages[i]
                        task_number = getattr(meta_msg, 'task_number', None)
                        msg_data["hierarchy"] = task_number.convert_to_str() if task_number else "unknown"
                        msg_data["status"] = getattr(meta_msg, 'status', 'unknown')
                        msg_data["type"] = getattr(meta_msg, 'type', 'unknown')

                removed_messages.append(msg_data)

            entry["removed_messages"] = removed_messages

            # Дополнительная информация о том, какие типы сообщений были удалены
            roles_summary = {}
            for msg in removed_messages:
                role = msg.get("role", "unknown")
                roles_summary[role] = roles_summary.get(role, 0) + 1

            entry["removed_roles_summary"] = roles_summary

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

            if self.console:
                roles_info = ", ".join([f"{count} {role}" for role, count in roles_summary.items()])

                self.console.print(f"[yellow]✂️ Обрезка контекста: удалено {removed_count} сообщений ({roles_info})[/]")

                if removed_count > 5:
                    self.console.print(f"[orange3]⚠️ Внимание: удалено значительное количество сообщений ({removed_count})[/]")

        except Exception as e:
            error_info = traceback.format_exc()
            if self.console:
                self.console.print(f"[bold red]Ошибка при логировании обрезки контекста: {e}[/]")

            # Логируем ошибку, но не прерываем выполнение
            error_entry = {
                "ts": datetime.now().isoformat(),
                "log_type": "trim_context_error",
                "error": str(e),
                "traceback": error_info
            }

            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    f.flush()
            except:
                pass

    def log_task_counter_state(self, depth: int = 0, extra: Dict[str, Any] = None) -> None:
        """
        Логирует текущее состояние TaskCounter для отслеживания иерархии задач.

        :param depth: Глубина рекурсии (для отображения).
        :param extra: Дополнительные метаданные для логирования.
        """
        if self.messages_meta_data is None or not hasattr(self.messages_meta_data, 'task_counter'):
            if self.console:
                self.console.print("[yellow]Не удалось логировать TaskCounter: отсутствует объект[/]")
            return

        task_counter = self.messages_meta_data.task_counter

        try:
            entry = {
                "ts": datetime.now().isoformat(),
                "log_type": "task_counter",
                "depth": depth,
                "hierarchy": task_counter.convert_to_str(),
                "task_order": task_counter.get_order(),
                "task_numbers": task_counter.numbers_array if hasattr(task_counter, 'numbers_array') else []
            }

            if extra:
                entry.update(extra)

            self._check_file_rotation()

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

            if self.console:
                self.console.print(f"[cyan]TaskCounter: {task_counter.convert_to_str()} (уровень: {task_counter.get_order()})[/]")
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Ошибка при логировании TaskCounter: {e}[/]")

    def log_context_to_file(self, file_name: Optional[str] = None) -> Optional[str]:
        """
        Сохраняет контекст модели в отдельный JSON-файл для более удобного просмотра.

        :param file_name: Имя файла для сохранения. Если None, используется текущая дата и время.
        :return: Путь к созданному файлу или None в случае ошибки.
        """
        if self.messages_meta_data is None or not hasattr(self.messages_meta_data, 'metadata_messages'):
            if self.console:
                self.console.print("[yellow]Не удалось сохранить контекст: отсутствуют метаданные сообщений[/]")
            return None

        try:
            if file_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"context_{timestamp}.json"

            log_dir = os.path.dirname(self.log_file)
            context_file_path = os.path.join(log_dir, file_name)

            context_data = {
                "timestamp": datetime.now().isoformat(),
                "task_counter": {
                    "value": self.messages_meta_data.task_counter.convert_to_str() if hasattr(self.messages_meta_data, 'task_counter') else "unknown",
                    "order": self.messages_meta_data.task_counter.get_order() if hasattr(self.messages_meta_data, 'task_counter') else -1,
                    "numbers": self.messages_meta_data.task_counter.numbers_array if hasattr(self.messages_meta_data, 'task_counter') and hasattr(self.messages_meta_data.task_counter, 'numbers_array') else []
                },
                "messages": []
            }

            for meta_msg in self.messages_meta_data.metadata_messages:
                task_number = meta_msg.task_number
                status = getattr(meta_msg, 'status', 'unknown')
                msg_type = getattr(meta_msg, 'type', 'unknown')
                message = getattr(meta_msg, 'message', {})

                role = message.get('role', 'unknown')
                content = self._extract_content(message.get('content', ''))

                message_data = {
                    "role": role,
                    "hierarchy": task_number.convert_to_str() if task_number else "unknown",
                    "task_order": task_number.get_order() if task_number else -1,
                    "status": status,
                    "type": msg_type,
                    "content": content,
                    "content_length": len(content)
                }

                context_data["messages"].append(message_data)

            with open(context_file_path, "w", encoding="utf-8") as f:
                json.dump(context_data, f, ensure_ascii=False, indent=2)

            if self.console:
                self.console.print(f"[bold green]Контекст успешно сохранен в файл: {context_file_path}[/]")

            return context_file_path
        except Exception as e:
            error_info = traceback.format_exc()
            if self.console:
                self.console.print(f"[bold red]Ошибка при сохранении контекста в файл: {e}[/]")
                self.console.print(f"[dim red]{error_info}[/]")
            return None

    def _check_file_rotation(self) -> None:
        """
        Проверяет размер файла и создает новый при превышении лимита.
        """
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_file_size:
                self.file_counter += 1
                self.log_file = f"{self.log_file_base}_{self.file_counter}.jsonl"
                if self.console:
                    self.console.print(f"[italic yellow]Создан новый лог-файл: {self.log_file}[/]")
        except Exception as e:
            # Если возникла ошибка, пытаемся создать новый файл в той же директории
            try:
                directory = os.path.dirname(self.log_file)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = os.path.join(directory, f"recursion_recovery_{timestamp}.jsonl")
                if self.console:
                    self.console.print(f"[bold red]Ошибка при проверке файла: {e}. Создан новый файл: {self.log_file}[/]")
            except:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = os.path.join(tempfile.gettempdir(), f"recursion_emergency_{timestamp}.jsonl")
                if self.console:
                    self.console.print(f"[bold red]Критическая ошибка с файловой системой. Используется временный файл: {self.log_file}[/]")

    def _get_hierarchy_id(self, depth: int, task_counter: Optional[TaskCounter] = None) -> str:
        """
        Создает иерархический ID вида '1.2.3', 'Исходная' или использует данные из TaskCounter.

        Приоритет нумерации:
        1. Если task_counter доступен, используется его значение (через convert_to_str)
        2. Если была сохранена иерархия для текущей фазы, используется она
        3. Если есть последний использованный hierarchy_id, используется он
        4. В крайнем случае, генерируется новый ID на основе depth_counters

        :param depth: Текущая глубина рекурсии.
        :param task_counter: Объект TaskCounter для получения иерархии задачи (если доступен).
        :return: Строка с иерархическим идентификатором (например, "1.2.3" или "Исходная").
        """
        # Если предоставлен task_counter, используем его как основной источник иерархии
        if task_counter is not None:
            try:
                hierarchy_id = task_counter.convert_to_str()

                if self.debug_numbering:
                    self.console.print(f"[dim green]_get_hierarchy_id: Используется TaskCounter: {hierarchy_id}[/]")

                return hierarchy_id
            except Exception as e:
                if self.debug_numbering:
                    self.console.print(f"[yellow]_get_hierarchy_id: ошибка TaskCounter: {str(e)}[/]")

        # Если есть last_hierarchy_id и нет других источников, используем его
        if self.last_hierarchy_id is not None:
            if self.debug_numbering:
                self.console.print(f"[dim yellow]_get_hierarchy_id: Используется last_hierarchy_id: {self.last_hierarchy_id}[/]")

            return self.last_hierarchy_id

        # Если все предыдущие источники недоступны, используем генерацию на основе depth
        # Пересчитываем счетчики глубины для текущего depth
        depths_to_reset = [d for d in self.depth_counters.keys() if d > depth]
        for d in depths_to_reset:
            del self.depth_counters[d]

        if depth not in self.depth_counters:
            self.depth_counters[depth] = 1
        else:
            self.depth_counters[depth] += 1

        # Если depth = 0, возвращаем "Исходная"
        if depth == 0:
            if self.debug_numbering:
                self.console.print(f"[dim yellow]_get_hierarchy_id: depth=0, иерархия='Исходная'[/]")
            return "Исходная"

        # Иначе формируем иерархию на основе depth_counters
        hierarchy = []
        for d in range(1, depth + 1):
            hierarchy.append(str(self.depth_counters.get(d, 1)))

        hierarchy_id = ".".join(hierarchy)

        if self.debug_numbering:
            self.console.print(f"[dim yellow]_get_hierarchy_id: Сгенерированная иерархия: {hierarchy_id}[/]")

        return hierarchy_id

    def _extract_content(self, content) -> str:
        """
        Извлекает текстовое содержимое из различных форматов сообщений.

        :param content: Содержимое сообщения (строка или список).
        :return: Извлеченный текст.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)

        return str(content)

    def _get_task_info_display(self, entry: Dict[str, Any]) -> Tuple[str, str]:
        """
        Формирует строки для отображения информации о задаче в консоли.

        :param entry: Словарь с данными записи.
        :return: Кортеж (hierarchy_display, meta_display) с строками для отображения.
        """
        hierarchy = entry.get('hierarchy', '')

        if hierarchy == "Исходная":
            hierarchy_display = "[Исходная]"
        else:
            hierarchy_display = f"[{hierarchy}]" if hierarchy else "[]"

        meta_info = []
        if "meta_type" in entry:
            meta_info.append(f"type=\"{entry['meta_type']}\"")
        if "meta_status" in entry:
            meta_info.append(f"status=\"{entry['meta_status']}\"")

        meta_display = f" [{', '.join(meta_info)}]" if meta_info else ""

        return hierarchy_display, meta_display

    def _print_to_console(self, entry: dict) -> None:
        """
        Выводит отформатированную запись лога в консоль.

        :param entry: Словарь с данными записи.
        """
        if not self.console:
            return

        emoji, color = self.phase_styles.get(entry["phase"], ("⚙️", "white"))
        indent = "   " * entry['depth']

        hierarchy_display, meta_display = self._get_task_info_display(entry)

        header = f"{indent}├── {emoji} {hierarchy_display} {entry['phase']}{meta_display}"
        self.console.print(header, style=f"bold {color}")

        if entry.get("error", False):
            error_text = f"{indent}│  [bold red]{entry['prompt']}[/]"
            self.console.print(error_text)
            if "context" in entry:
                context_text = f"{indent}│  [dim]{entry['context']}[/]"
                self.console.print(context_text)
            return

        console_width = self.console.width

        indent_length = len(indent) + 3  # +3 для "│  "
        available_width = console_width - indent_length

        if "prompt_preview" in entry:
            text = entry['prompt_preview']
            # text = entry['prompt']
            wrapped_lines = textwrap.wrap(text, width=available_width) + ["⎯⎯⎯"]
            for line in wrapped_lines:
                formatted_line = f"{indent}│  {line}"
                self.console.print(formatted_line, style="dim")

        if "response_preview" in entry:
            # text = entry['response_preview']
            text = entry['response']
            wrapped_lines = textwrap.wrap(text, width=available_width)
            for line in wrapped_lines:
                formatted_line = f"{indent}│  {line}"
                self.console.print(formatted_line, style="dim italic")
