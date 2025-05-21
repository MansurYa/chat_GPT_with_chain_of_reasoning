from typing import List, Dict, Optional
import copy
import re
import logging
import traceback

from src.utils import TaskCounter, dedent_text, add_indent


class MessageMetaData:
    """
    Класс для хранения мета-информации о сообщении.
    """
    def __init__(self, task_counter: TaskCounter, status: str, message_type: str, message: Dict):
        self.task_number = copy.deepcopy(task_counter)
        self.status = status
        self.type = message_type
        self.message = message

    def convert_metadata_to_string(self):
        return (
            "Задача "
            + self.task_number.convert_to_str()
            + f" [status=\"{self.status}\""
            + f" type=\"{self.type}\"]"
        )


class MessagesWithMetaData:
    def __init__(self, messages: List[Dict]):
        self.messages: List[Dict] = messages
        self.metadata_messages: List[MessageMetaData] = []
        self.task_counter = TaskCounter()

    def add_metadata_in_last_message(self, status: str, message_type: str, command_number: int):
        """
        command_number == 0 - номер задачи остаётся тот же
        command_number == 1 - номер задачи увеличился на одну единицу
        command_number == 2 - глубина задачи увеличилась на один уровень
        command_number == 3 - глубина задачи уменьшилась на один уровень
        """
        if command_number < 0 or command_number > 3:
            raise Exception("Недопустимый номер команды для взаимодействия с TaskCounter")

        if not self.messages:
            raise Exception("Контекст не может быть пустым")

        if command_number == 1:
            self.task_counter.increase_digit()
        elif command_number == 2:
            self.task_counter.increase_order()
        elif command_number == 3:
            self.task_counter.reduce_order()

        # Добавляем метаданные
        self.metadata_messages.append(
            MessageMetaData(
                task_counter=self.task_counter,
                status=status,
                message_type=message_type,
                message=self.messages[-1]
            )
        )

        # Получаем текущий контент
        current_content = self.messages[-1]["content"]

        # Создаем метаданные и форматируем текст
        metadata_string = self.metadata_messages[-1].convert_metadata_to_string()
        indent_level = self.metadata_messages[-1].task_number.get_order()

        # Обрабатываем в зависимости от типа content
        if isinstance(current_content, list):
            # Для списка (мультимодальный формат)
            for item in current_content:
                if item.get("type") == "text":
                    # Извлекаем текст, обрабатываем и обновляем
                    text = item["text"]
                    text = dedent_text(text)
                    text = metadata_string + "\n" + text
                    text = add_indent(text, indent_level)
                    item["text"] = text
                    break
            else:
                # Если текстовый элемент не найден, добавляем новый в начало
                formatted_text = metadata_string + "\n"
                formatted_text = add_indent(formatted_text, indent_level)
                current_content.insert(0, {"type": "text", "text": formatted_text})
        else:
            # Для строки (старый формат)
            text = current_content
            text = dedent_text(text)
            text = metadata_string + "\n" + text
            text = add_indent(text, indent_level)
            self.messages[-1]["content"] = text


    def update_all_messages_statuses(self):
        """
        Пересчитывает и проставляет статусы для всех сообщений, исходя из того,
        что последняя задача считается 'in_progress', а остальные распределяются
        по логике ancestor/descendant/другие ветки.
        """

        if not self.metadata_messages:
            return

        # Определяем номер задачи последнего (самого свежего) сообщения — считаем её текущей (in_progress)
        current_task_array = self.metadata_messages[-1].task_number.numbers_array

        def is_ancestor(anc, desc):
            """Проверка: является ли anc префиксом desc. Пустой массив (Исходная) тоже считается предком любого."""
            if len(anc) > len(desc):
                return False
            for i in range(len(anc)):
                if anc[i] != desc[i]:
                    return False
            return True

        for meta_msg in self.metadata_messages:
            arr = meta_msg.task_number.numbers_array

            if arr == current_task_array:
                # Это текущая задача
                new_status = "in_progress"
            elif is_ancestor(arr, current_task_array):
                # Предок текущей задачи (включая "Исходная", если arr пуст)
                new_status = "parent_for_current_task"
            elif is_ancestor(current_task_array, arr):
                # Потомок (подзадача) текущей задачи
                # Считаем её "subtask_resolved", т.к. текущая задача "in_progress"
                new_status = "subtask_resolved"
            else:
                # Если верхний уровень совпадает, значит это соседняя задача -> "resolved"
                # Иначе, считаем её "resolved_subtask_of_parent_not_important_for_current"
                if arr and current_task_array and arr[0] == current_task_array[0]:
                    new_status = "resolved"
                else:
                    new_status = "resolved_subtask_of_parent_not_importante_for_current"
                # Исправим опечатку (importante -> important)
                if new_status == "resolved_subtask_of_parent_not_importante_for_current":
                    new_status = "resolved_subtask_of_parent_not_important_for_current"

            meta_msg.status = new_status

    def rewrite_messages_content_with_updated_statuses(self):
        """
        Проходится по всем сообщениям и, если у них внутри content есть в метаданных
        status="...", заменяет его на новый (meta_msg.status).
        """
        pattern = r'(status=")([^"]*)(")'
        for meta_msg in self.metadata_messages:
            content = meta_msg.message["content"]

            if isinstance(content, list):
                # Для списка (мультимодальный формат)
                for item in content:
                    if item.get("type") == "text" and "status=" in item.get("text", ""):
                        text = item["text"]
                        match = re.search(pattern, text)
                        if match and match.group(2) != meta_msg.status:
                            # Заменяем статус и применяем отступы
                            new_text = re.sub(pattern, rf'\1{meta_msg.status}\3', text)
                            item["text"] = add_indent(new_text, meta_msg.task_number.get_order())
                        break
            else:
                # Для строки (старый формат)
                match = re.search(pattern, content)
                if match and match.group(2) != meta_msg.status:
                    # Заменяем статус и применяем отступы
                    new_content = re.sub(pattern, rf'\1{meta_msg.status}\3', content)
                    meta_msg.message["content"] = add_indent(
                        new_content, meta_msg.task_number.get_order()
                    )

    def clone(self, new_messages_list: List[Dict]) -> 'MessagesWithMetaData':
        """
        Создает полную копию объекта MessagesWithMetaData, связанную с новым списком сообщений.

        :param new_messages_list: Новый список сообщений из клонированного контекста
        :return: Клонированный экземпляр MessagesWithMetaData
        """
        STATUS_PATTERN = re.compile(r'status="([^"]+)"')
        TYPE_PATTERN = re.compile(r'type="([^"]+)"')

        # Создаем новый экземпляр
        cloned = MessagesWithMetaData(new_messages_list)

        # Копируем счетчик задач для сохранения иерархии
        cloned.task_counter = copy.deepcopy(self.task_counter)

        # Если нет метаданных или сообщений, просто возвращаем базовый клон
        if not self.metadata_messages or not new_messages_list or not self.messages:
            return cloned

        # Логируем предупреждение при несовпадении количества сообщений
        if len(new_messages_list) != len(self.messages):
            logging.warning(
                f"Количество сообщений в клонированном объекте ({len(new_messages_list)}) "
                f"не соответствует исходному ({len(self.messages)}). "
                f"Метаданные могут быть восстановлены не полностью."
            )

        try:
            # Создаем словарь исходных метаданных по сигнатуре сообщения
            metadata_dict = {}
            for meta in self.metadata_messages:
                if meta.message is None:
                    continue

                # Создаем уникальную сигнатуру для сообщения
                content = meta.message.get("content", "")
                role = meta.message.get("role", "")

                if isinstance(content, list):
                    # Для мультимодального контента используем первый текстовый элемент
                    content_key = ""
                    for item in content:
                        if item.get("type") == "text":
                            content_key = item.get("text", "")[:50]
                            break
                else:
                    content_key = str(content)[:50]

                # Формируем ключ из роли и начала контента
                signature = f"{role}:{content_key}"
                metadata_dict[signature] = meta

            # Массив для отслеживания уже добавленных метаданных
            processed_indices = []

            # Обработка исходных сообщений
            # Поскольку add_metadata_in_last_message работает только с последним сообщением,
            # мы будем добавлять сообщения и метаданные в правильном порядке

            # 1. Сначала сохраняем исходные сообщения из cloned
            original_messages = list(cloned.messages)

            # 2. Очищаем список сообщений
            cloned.messages.clear()

            # 3. Последовательно добавляем сообщения с метаданными
            for i, msg in enumerate(original_messages):
                # Добавляем сообщение в список
                cloned.messages.append(msg)

                # Создаем сигнатуру для поиска метаданных
                content = msg.get("content", "")
                role = msg.get("role", "")

                content_key = ""
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            content_key = item.get("text", "")[:50]
                            break
                else:
                    content_key = str(content)[:50]

                signature = f"{role}:{content_key}"

                # Если есть метаданные - используем их
                meta_found = False
                if signature in metadata_dict:
                    meta = metadata_dict[signature]
                    # Добавляем метаданные
                    cloned.add_metadata_in_last_message(
                        status=meta.status,
                        message_type=meta.type,
                        command_number=0
                    )

                    # Копируем пользовательские атрибуты
                    for attr_name, attr_value in vars(meta).items():
                        if attr_name not in ['task_number', 'status', 'type', 'message']:
                            try:
                                setattr(cloned.metadata_messages[-1], attr_name,
                                       copy.deepcopy(attr_value))
                            except (TypeError, AttributeError):
                                # Для объектов, которые нельзя скопировать
                                setattr(cloned.metadata_messages[-1], attr_name, attr_value)

                    processed_indices.append(i)
                    meta_found = True

                # Если метаданные не найдены, попробуем восстановить из текстовых меток
                if not meta_found:
                    status = "unknown"
                    msg_type = "unknown"

                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                status_match = STATUS_PATTERN.search(text)
                                type_match = TYPE_PATTERN.search(text)
                                if status_match and type_match:
                                    status = status_match.group(1)
                                    msg_type = type_match.group(1)
                                    meta_found = True
                                    break
                    elif isinstance(content, str):
                        status_match = STATUS_PATTERN.search(content)
                        type_match = TYPE_PATTERN.search(content)
                        if status_match and type_match:
                            status = status_match.group(1)
                            msg_type = type_match.group(1)
                            meta_found = True

                    if meta_found:
                        cloned.add_metadata_in_last_message(
                            status=status,
                            message_type=msg_type,
                            command_number=0
                        )
                        processed_indices.append(i)

        except Exception as e:
            logging.error(f"Ошибка при реконструкции метаданных: {str(e)}")
            # Удалено использование traceback

        return cloned

    # Далее идёт 6 функций, которые нужны для замены на сокращённые промпты.
    def find_messages_by_type(self, message_type: str) -> list:
        """
        Находит все метаданные сообщений с указанным типом в истории сообщений.

        :param message_type: Тип сообщения для поиска (например, "Theory", "Solution")
        :return: Список объектов MessageMetaData, соответствующих указанному типу
        """
        if not message_type or not isinstance(message_type, str):
            logging.warning(f"Некорректный тип сообщения для поиска: {message_type}")
            return []

        if not hasattr(self, 'metadata_messages') or not self.metadata_messages:
            logging.warning("Список metadata_messages пуст или отсутствует")
            return []

        matching_messages = []
        for meta_msg in self.metadata_messages:
            if hasattr(meta_msg, 'type') and meta_msg.type == message_type:
                matching_messages.append(meta_msg)

        logging.debug(f"Найдено {len(matching_messages)} сообщений типа '{message_type}'")
        return matching_messages


    def _is_prompt_already_shortened(self, meta_msg) -> bool:
        """
        Проверяет, был ли промпт уже сокращен.

        :param meta_msg: Объект MessageMetaData
        :return: True, если промпт уже был сокращен, иначе False
        """
        # Защита от None
        if meta_msg is None or not hasattr(meta_msg, "message") or meta_msg.message is None:
            logging.debug("Пустой объект meta_msg при проверке сокращенного промпта")
            return False

        # Проверяем наличие специальной метки
        if hasattr(meta_msg, "shortened") and meta_msg.shortened:
            return True

        # Получаем содержимое с защитой от None
        content = meta_msg.message.get("content")
        if content is None:
            return False

        # Эвристическая проверка длины
        if isinstance(content, list):
            # Обработка мультимодального содержимого
            text_items = [item for item in content if item.get("type") == "text" and "text" in item]
            if not text_items:
                return False

            # Проверяем первый текстовый элемент
            text = text_items[0].get("text", "")
            lines = text.splitlines()
            if len(lines) > 1:
                prompt_content = "\n".join(lines[1:])
                # Если длина промпта меньше 300 символов, считаем его уже сокращенным
                if len(prompt_content.strip()) < 300:
                    logging.debug(f"Промпт типа '{meta_msg.type}' считается сокращенным по длине")
                    return True

        elif isinstance(content, str):
            # Обработка текстового содержимого
            lines = content.splitlines()
            if len(lines) > 1:
                prompt_content = "\n".join(lines[1:])
                if len(prompt_content.strip()) < 300:
                    logging.debug(f"Промпт типа '{meta_msg.type}' считается сокращенным по длине")
                    return True

        return False


    def _mark_prompt_as_shortened(self, meta_msg) -> None:
        """
        Отмечает промпт как сокращенный.

        :param meta_msg: Объект MessageMetaData
        """
        # Защита от None
        if meta_msg is None:
            logging.warning("Попытка отметить None объект как сокращенный промпт")
            return

        # Добавляем атрибут для отслеживания
        setattr(meta_msg, "shortened", True)
        logging.debug(f"Промпт типа '{meta_msg.type if hasattr(meta_msg, 'type') else 'unknown'}' отмечен как сокращенный")


    def replace_prompt_in_message(self, meta_msg, shortened_prompt: str) -> bool:
        """
        Заменяет содержимое промпта в указанном сообщении, сохраняя метаданные.

        :param meta_msg: Объект MessageMetaData, указывающий на сообщение для замены
        :param shortened_prompt: Сокращенная версия промпта
        :return: True если операция успешна, False в противном случае
        """
        try:
            # Защита от None
            if meta_msg is None or not hasattr(meta_msg, "message") or meta_msg.message is None:
                logging.warning("Пустой объект meta_msg при замене промпта")
                return False

            original_message = meta_msg.message
            content = original_message.get("content")

            if content is None:
                logging.warning("Пустое содержимое сообщения при замене промпта")
                return False

            # Обработка мультимодального содержимого (список объектов)
            if isinstance(content, list):
                # Находим все текстовые элементы
                text_items = [item for item in content if item.get("type") == "text" and "text" in item]
                if not text_items:
                    logging.warning("Не найдены текстовые элементы в мультимодальном сообщении")
                    return False

                # Обрабатываем первый текстовый элемент
                item = text_items[0]
                text = item.get("text", "")

                # Извлечение первой строки с метаданными
                lines = text.splitlines(False)  # Не сохраняем символы переноса строки
                if not lines:
                    logging.warning("Пустое текстовое содержимое в мультимодальном сообщении")
                    return False

                # Проверка наличия метаданных
                metadata_line = lines[0]
                if "status=" not in metadata_line or "type=" not in metadata_line:
                    logging.warning(f"Возможно, строка не содержит метаданные: {metadata_line}")

                # Составляем новый текст: метаданные + сокращенный промпт
                new_text = metadata_line + "\n" + shortened_prompt

                # Применяем отступ в соответствии с уровнем задачи
                indent_level = meta_msg.task_number.get_order() if hasattr(meta_msg, "task_number") else 0
                item["text"] = add_indent(new_text, indent_level)
                return True

            # Обработка текстового содержимого (строка)
            elif isinstance(content, str):
                lines = content.splitlines(False)  # Не сохраняем символы переноса строки
                if not lines:
                    logging.warning("Пустое текстовое содержимое в сообщении")
                    return False

                # Проверка наличия метаданных
                metadata_line = lines[0]
                if "status=" not in metadata_line or "type=" not in metadata_line:
                    logging.warning(f"Возможно, строка не содержит метаданные: {metadata_line}")

                # Составляем новый текст: метаданные + сокращенный промпт
                new_content = metadata_line + "\n" + shortened_prompt

                # Применяем отступ в соответствии с уровнем задачи
                indent_level = meta_msg.task_number.get_order() if hasattr(meta_msg, "task_number") else 0
                original_message["content"] = add_indent(new_content, indent_level)
                return True

            logging.warning(f"Неизвестный формат содержимого: {type(content)}")
            return False
        except Exception as e:
            error_traceback = traceback.format_exc()
            logging.error(f"Ошибка при замене промпта: {str(e)}\n{error_traceback}")
            return False


    def replace_prompts_by_type(self, message_type: str, shortened_prompt: str, strategy: str = "all") -> int:
        """
        Заменяет промпты указанного типа их сокращенными версиями.

        :param message_type: Тип сообщения для замены
        :param shortened_prompt: Сокращенная версия промпта
        :param strategy: Стратегия замены - "all" (все), "last" (только последний), "first" (только первый)
        :return: Количество замененных промптов
        """
        matching_messages = self.find_messages_by_type(message_type)
        if not matching_messages:
            logging.warning(f"Промпты типа '{message_type}' не найдены")
            return 0

        # Применяем выбранную стратегию с защитой от пустого списка
        if strategy == "last" and matching_messages:
            matching_messages = [matching_messages[-1]]
        elif strategy == "first" and matching_messages:
            matching_messages = [matching_messages[0]]
        elif strategy != "all":
            logging.warning(f"Неизвестная стратегия замены '{strategy}'. Используется 'all'")

        replaced_count = 0
        for meta_msg in matching_messages:
            # Проверяем, был ли промпт уже заменен
            if self._is_prompt_already_shortened(meta_msg):
                logging.debug(f"Промпт типа '{message_type}' уже был сокращен")
                continue

            if self.replace_prompt_in_message(meta_msg, shortened_prompt):
                # Отмечаем, что промпт был заменен
                self._mark_prompt_as_shortened(meta_msg)
                replaced_count += 1

        logging.info(f"Заменено {replaced_count} промптов типа '{message_type}'")
        return replaced_count


    def safe_replace_prompt(self, message_type: str, shortened_prompt: str,
                            # debug_tracer: Optional[DebugTracer] = None,
                            debug_tracer=None,
                            depth: int = 0,
                            strategy: str = "all") -> bool:
        """
        Безопасно заменяет промпты с обработкой ошибок и логированием.

        :param message_type: Тип сообщения для замены
        :param shortened_prompt: Сокращенная версия промпта
        :param debug_tracer: Объект DebugTracer для логирования
        :param depth: Текущая глубина рекурсии для логирования
        :param strategy: Стратегия замены - "all", "last" или "first"
        :return: True если замена успешна, иначе False
        """
        try:
            if not message_type or not isinstance(message_type, str):
                logging.warning(f"Некорректный тип сообщения для замены: {message_type}")
                return False

            if not shortened_prompt or not isinstance(shortened_prompt, str):
                logging.warning("Пустая или некорректная сокращенная версия промпта")
                return False

            replaced_count = self.replace_prompts_by_type(message_type, shortened_prompt, strategy)

            # if debug_tracer:
            #     debug_tracer.log(
            #         depth=depth,
            #         phase="Prompt Optimization",
            #         prompt=f"Замена промптов типа '{message_type}'",
            #         extra={"type": message_type, "replaced": replaced_count, "strategy": strategy}
            #     )

            return replaced_count > 0
        except Exception as e:
            error_traceback = traceback.format_exc()
            logging.error(f"Ошибка при безопасной замене промптов: {str(e)}\n{error_traceback}")

            if debug_tracer:
                debug_tracer.log_error(
                    depth=depth,
                    error_msg=f"Ошибка замены промптов типа '{message_type}'",
                    context=f"{str(e)}\n{error_traceback}"
                )

            return False
