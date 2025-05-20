import os
import copy
import base64


class MessageContext:
    """
    Класс MessageContext управляет добавлением сообщений в контекст для LLM, поддерживая три режима работы:

    Режимы (моды):
        1. Очищает контекст перед каждым добавлением нового пользовательского сообщения. В пустой контекст добавляется task_prompt и сообщение пользователя.
        2. Сохраняет контекст между запросами. task_prompt добавляется только один раз в самое начало, после чего каждое новое сообщение добавляется поверх предыдущих. При изменении task_prompt добавляется новое системное сообщение.
        3. Сохраняет контекст между запросами. task_prompt добавляется перед каждым пользовательским сообщением, но не перед сообщениями ассистента.

    Класс также поддерживает подсчёт токенов, обрезку контекста и обновление task_prompt для дальнейшего использования.
    """

    def __init__(self, mode: int, task_prompt: str = None):
        """
        Инициализирует экземпляр класса MessageContext.

        :param mode: Режим работы (1, 2 или 3).
        :param task_prompt: Начальное задание, добавляемое в контекст.
        :raises ValueError: Если mode не равен 1, 2 или 3.
        """
        if mode not in [1, 2, 3]:
            raise ValueError("Неверный режим. Режим должен быть 1, 2 или 3.")
        self.mode = mode
        self.task_prompt = task_prompt
        self.messages = []

    def change_mod(self, new_mode):
        """
        Изменяет мод добавления сообщений.

        :param new_mode: Режим работы (1, 2 или 3).
        :raises ValueError: Если mode не равен 1, 2 или 3.
        """
        if new_mode not in [1, 2, 3]:
            raise ValueError("Неверный режим. Режим должен быть 1, 2 или 3.")

        self.mode = new_mode

    def brutally_convert_to_message(self, role: str, text: str, images: list = None):
        """
        Конвертирует сообщение в формат для MessageContext.messages и возвращает его.
        Поддерживает добавление текста и изображений.

        :param role: Роль сообщения ("user" или "assistant").
        :param text: Текст сообщения.
        :param images: Список изображений (URL или локальные файлы).
        :return: Словарь с сообщением в нужном формате
        """
        content = [{"type": "text", "text": text}]

        if images is not None:
            for image in images:
                if self.__is_url(image):
                    if any(image.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "low"}
                        })
                elif os.path.isfile(image):  # если локальный путь
                    base64_image = self.__encode_image_to_base64(image)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                        })

        # Добавляем сообщение в список независимо от режима
        return {"role": role, "content": content}

    def update_task_prompt(self, new_task_prompt: str):
        """
        Обновляет task_prompt для добавления в контекст в дальнейшем.

        :param new_task_prompt: Новый task_prompt.
        """
        self.task_prompt = new_task_prompt
        # Если режим == 2, добавляем новый task_prompt как системное сообщение, не удаляя первое сообщение
        if self.mode == 2 and len(self.messages) > 0:
            self.messages.append({"role": "system", "content": self.task_prompt})

    def add_user_message(self, text: str, images: list = None):
        """
        Добавляет пользовательское сообщение в контекст с учетом режима работы.

        :param text: Текст сообщения.
        :param images: Опциональный список изображений (URL или локальные файлы). Если указаны, изображения добавляются к сообщению с низким уровнем детализации ("low").
        """
        # Формируем контент сообщения
        content = [{"type": "text", "text": text}]

        # Обрабатываем изображения
        if images is not None:
            for image in images:
                if self.__is_url(image):
                    if any(image.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "low"}
                        })
                elif os.path.isfile(image):  # если локальный путь
                    base64_image = self.__encode_image_to_base64(image)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                        })

        # Добавляем сообщение в зависимости от выбранного режима
        if self.mode == 1:
            self.__add_message_mode_1("user", content)
        elif self.mode == 2:
            self.__add_message_mode_2("user", content)
        elif self.mode == 3:
            self.__add_message_mode_3("user", content)

    def __is_url(self, image: str) -> bool:
        """
        Проверяет, является ли строка URL-адресом изображения.

        :param image: Предполагаемый URL изображения
        :return: True, если это URL, иначе False
        """
        return image.startswith("http://") or image.startswith("https://")

    def __encode_image_to_base64(self, image_path: str) -> str:
        """
        Кодирует изображение из локального пути в base64 формат.

        :param image_path: Путь к локальному файлу изображения
        :return: Закодированная в base64 строка или пустая строка в случае ошибки
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Ошибка при кодировании изображения: {e}")

    def add_assistant_message(self, text: str, images: list = None):
        """
        Добавляет сообщение ассистента в контекст в соответствии с режимом.

        :param text: Текст сообщения.
        :param images: Список изображений (URL или локальные файлы).
        """
        # Формируем контент сообщения
        content = [{"type": "text", "text": text}]

        # Обрабатываем изображения
        if images is not None:
            for image in images:
                if self.__is_url(image):
                    if any(image.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "low"}
                        })
                elif os.path.isfile(image):  # если локальный путь
                    base64_image = self.__encode_image_to_base64(image)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                        })

        # Добавляем сообщение в зависимости от выбранного режима
        if self.mode == 1:
            self.__add_message_mode_1("assistant", content)
        elif self.mode == 2:
            self.__add_message_mode_2("assistant", content)
        elif self.mode == 3:
            self.__add_message_mode_3("assistant", content)

    def __add_message_mode_1(self, role: str, content: list):
        """
        Добавляет сообщение в режиме 1.

        :param role: Роль сообщения ("user" или "assistant").
        :param content: Список с текстом и изображениями.
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Неверный тип пользователя. Должен быть 'user' или 'assistant'.")

        # Если роль - "user", очищаем контекст и добавляем task_prompt и новое сообщение
        if role == "user" and self.task_prompt:
            self.messages = [{"role": "system", "content": [{"type": "text", "text": self.task_prompt}]},
                             {"role": role, "content": content}]
        elif role == "user":    # self.task_prompt - отсутствует
            self.messages = [{"role": role, "content": content}]
        elif role == "assistant":
            # Если роль - "assistant", просто добавляем сообщение в конец
            self.messages.append({"role": role, "content": content})

    def __add_message_mode_2(self, role: str, content: list):
        """
        Добавляет сообщение в режиме 2.

        :param role: Роль сообщения ("user" или "assistant").
        :param content: Список с текстом и изображениями.
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Неверный тип пользователя. Должен быть 'user' или 'assistant'.")

        # Если сообщений еще нет и task_prompt задан, добавляем task_prompt, но только один раз
        if self.task_prompt and len(self.messages) == 0:
            self.messages.append({"role": "system", "content": [{"type": "text", "text": self.task_prompt}]})

        # Добавляем новое сообщение поверх предыдущих
        self.messages.append({"role": role, "content": content})

    def __add_message_mode_3(self, role: str, content: list):
        """
        Добавляет сообщение в режиме 3.

        :param role: Тип пользователя ("user" или "assistant").
        :param content: Список с текстом и изображениями.
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Неверный тип пользователя. Должен быть 'user' или 'assistant'.")

        # Если роль - "user", добавляем task_prompt перед каждым новым пользовательским сообщением
        if role == "user" and self.task_prompt:
            self.messages.append({"role": "system", "content": [{"type": "text", "text": self.task_prompt}]})
            self.messages.append({"role": role, "content": content})
        elif role == "user":    # task_prompt - отсутствует
            self.messages.append({"role": role, "content": content})
        elif role == "assistant":
            # Если роль - "assistant", добавляем сообщение в конец контекста без task_prompt
            self.messages.append({"role": role, "content": content})

    def get_message_history(self) -> list:
        """
        Возвращает копию списка сообщений для работы с ним без изменения оригинального контекста.

        :return: Копия списка сообщений.
        """
        return self.messages.copy()

    def clone(self):
        """
        Создает глубокую копию текущего объекта MessageContext, включая историю сообщений и все параметры.

        :return: Новый экземпляр MessageContext с идентичными параметрами и историей.
        """
        # Создаем новый объект MessageContext с теми же параметрами и task_prompt
        cloned_context = MessageContext(self.mode, self.task_prompt)

        # Копируем историю сообщений и все атрибуты
        cloned_context.messages = copy.deepcopy(self.messages)

        return cloned_context
