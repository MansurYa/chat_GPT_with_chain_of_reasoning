from openai import OpenAI, RateLimitError, APITimeoutError, APIError
import tiktoken
import os
import copy
import base64
from typing import Optional, Union, Type
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


class MessageContext:
    """
    Класс MessageContext управляет добавлением сообщений в контекст для чата GPT, поддерживая три режима работы:

    Режимы (моды):
        1. Очищает контекст перед каждым добавлением нового пользовательского сообщения. В пустой контекст добавляется task_prompt и сообщение пользователя.
        2. Сохраняет контекст между запросами. task_prompt добавляется только один раз в самое начало, после чего каждое новое сообщение добавляется поверх предыдущих. При изменении task_prompt добавляется новое системное сообщение.
        3. Сохраняет контекст между запросами. task_prompt добавляется перед каждым пользовательским сообщением, но не перед сообщениями ассистента.

    Класс также поддерживает подсчёт токенов, обрезку контекста и обновление task_prompt для дальнейшего использования.
    """

    def __init__(self, mode: int, task_prompt: str):
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

        Описание работы:
        - Режим 1: Очищает контекст, добавляет task_prompt и сообщение.
        - Режим 2: Сохраняет task_prompt один раз и добавляет сообщение поверх предыдущих.
        - Режим 3: Добавляет task_prompt перед каждым новым пользовательским сообщением, не дублируя перед сообщениями ассистента.

        Обработка изображений:
        - URL с корректным расширением (.jpg, .jpeg, .png, .gif, .webp) или локальный файл, который кодируется в Base64.
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
        """Проверяет, является ли строка URL-адресом изображения."""
        return image.startswith("http://") or image.startswith("https://")

    def __encode_image_to_base64(self, image_path: str) -> str:
        """Кодирует изображение из локального пути в base64 формат."""
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
        if role == "user":
            self.messages = [{"role": "system", "content": [{"type": "text", "text": self.task_prompt}]},
                             {"role": role, "content": content}]
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

        # Если сообщений еще нет, добавляем task_prompt только один раз
        if len(self.messages) == 0:
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
        if role == "user":
            self.messages.append({"role": "system", "content": [{"type": "text", "text": self.task_prompt}]})
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


class ChatGPTAgent:
    """
    Класс ChatGPTAgent взаимодействует с API chat GPT, используя MessageContext для управления контекстом сообщений.
    """

    def __init__(self, api_key: str, organization: str, model_name: str, mode: int, task_prompt: str,
                 max_total_tokens: int = 32000, max_response_tokens: int = 4095, temperature: float = 0.0):
        """
        Инициализирует агент с API-ключом и настраивает контекст сообщений.

        Режимы (моды):
            1. Очищает контекст перед каждым добавлением нового пользовательского сообщения. В пустой контекст добавляется task_prompt и сообщение пользователя.
            2. Сохраняет контекст между запросами. task_prompt добавляется только один раз в самое начало, после чего каждое новое сообщение добавляется поверх предыдущих. При изменении task_prompt добавляется новое системное сообщение.
            3. Сохраняет контекст между запросами. task_prompt добавляется перед каждым пользовательским сообщением, но не перед сообщениями ассистента.

        :param api_key: API-ключ для доступа к chat GPT.
        :param organization: Organization ID для доступа к chat GPT.
        :param model_name: Название модели для API OpenAI.
        :param mode: Режим работы контекста.
        :param task_prompt: Задание, добавляемое в контекст.
        :param max_total_tokens: Максимальное количество токенов для запроса.
        :param max_response_tokens: Максимальное количество токенов в ответе.
        :param temperature: Параметр температуры для API (креативность ответов).
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_total_tokens = max_total_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

        self.client = OpenAI(
            organization=organization,
            api_key=api_key
        )

        self.context = MessageContext(mode=mode, task_prompt=task_prompt)

    def response_from_chat_GPT(self, user_message: str, images: list = None, response_format: Optional[Type[BaseModel]] = None) -> str:
        """
        Добавляет сообщение пользователя, получает ответ от чата GPT и добавляет его в контекст.

        :param user_message: Сообщение пользователя для добавления в контекст и отправки в API.
        :param images: Список изображений (если есть).
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :return: Ответ ассистента.
        """
        # Добавляем пользовательское сообщение в контекст
        self.context.add_user_message(user_message, images)

        # Получаем копию контекста и обрезаем её
        messages = self.context.get_message_history()
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        # Вызываем API и получаем ответ
        assistant_response = self.__call_openai_api(trimmed_messages, response_format)

        # Проверяем, удалось ли получить ответ
        if assistant_response is None:
            print("Ошибка: ответ от API не был получен для response_from_chat_GPT.")
            return "Ошибка: не удалось получить ответ от API."

        # Добавляем ответ ассистента в контекст
        self.context.add_assistant_message(assistant_response)
        return assistant_response

    # TO-DO:
    #     1. analysis_depth: int = None - сделать автоматическое определение analysis_depth.
    #     2. Сейчас вопросы задаются лишь в начале.
    #        А можно было бы процесс рассуждения разбить на раунды, в каждом из которых задаются новые вопросы.
    def response_from_chat_GPT_with_chain_of_reasoning(self, analysis_depth: int, user_message: str, images: list = None, preserve_user_messages_post_analysis: bool = True, response_format: Optional[Type[BaseModel]] = None, debug_reasoning_print=False) -> str:
        """
        Делает то же самое, что и response_from_chat_GPT, но с использованием метода цепочки рассуждений.
        Она позволяет глубже анализировать запросы, формируя структурированный и обоснованный ответ.

        :param analysis_depth: Определяет количество итераций для анализа запроса. Чем больше значение, тем более детальный анализ будет проведен.
        :param user_message: Сообщение от пользователя, которое требует анализа и ответа.
        :param images: Список изображений, связанных с пользовательским сообщением (список путей до изображений). По умолчанию None.
        :param preserve_user_messages_post_analysis: Указывает, следует ли сохранять входящее пользовательское сообщения после анализа. По умолчанию True.
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :param debug_reasoning_print: Вывод отладочной информации в консоль.
        :return: Ответ ассистента.
        """
        if analysis_depth < 1:
            raise Exception("Ошибка: глубина анализа должна быть не менее 1. Пожалуйста, укажите корректное значение для analysis_depth.")

        if preserve_user_messages_post_analysis:
            self.context.add_user_message(user_message, images)

        if debug_reasoning_print:
            print(f"Запуск response_from_chat_GPT_with_chain_of_reasoning\nВходное сообщение:\n{user_message}\nВходные изображения: {images}")

        saved_context = self.context
        self.context = self.context.clone()

        self.context.change_mod(2)
        self.context.add_user_message(f"""
!Мы вошли в режим глубоких рассуждений!
Сейчас я говорю от лица администратора:
        
Инструкции: 
    Если тебя просят перевести текст, то переводи не просто дословно, а передавай смысл, который был заложен автором, делая перевод профессионально ориентированным. Перевод должен быть доступным для читателя с техническим фоном и сохранять сложные обороты, если они несут важный смысл. При переводе технических текстов, если встречаются профессиональные термины на английском языке, не переводи их на русский, а оставляй на английском. Если есть русский аналог термина, то вставляй его в скобках перед английским термином. Ты имеешь большой опыт работы в сфере IT, что позволяет эффективно переводить сложные технические тексты, особенно по базам данных и распределённым вычислениям.
    Если вопросы по высшей математике, представь, что ты преподаватель математического анализа. Помогай разбираться с темами пошагово, строго придерживаясь математической точности и приводя примеры с пошаговыми объяснениями. Разбирай возможные ошибки, чтобы предупредить неверные интерпретации, и объясняй каждое утверждение так, чтобы оно было доступным и математически точным.
                
Тебе пришло новое сообщение от пользователя:
```
{user_message}
f```
        """, images)

        self.context.add_user_message(f"""
Сейчас нужно сд:
1. Подумай над тем какой ответ от тебя ожидает пользователь в идеале?
2. Подумай что нужно прояснить, вычислить, написать, узнать, переписать, на что ответить, чтобы твой итоговый ответ пользователю был идеально правильным, без единой ошибки!
3. (главный) Мы скоро займёмся анализом задачи, которую поставил пользователь своим сообщением. Напиши дорожную карту для решения вопроса пользователя из {analysis_depth} пунктов, через которые тебе нужно пройти, чтобы в конце дать ответ пользователю, который будет без ошибок и с со всеми решёнными задачам, что были в этом сообщении.
    Каждый из {analysis_depth} пунктов, должен представлять собой: 
        - либо вопрос, на который тебе следует ответить перед тем как отвечать пользователю и который поможет тебе правильнее выполнить задачу пользователя
        - либо подзадача, которую нужно проанализировать и выполнить перед тем как отвечать пользователю, чтобы правильнее выполнить задачу пользователя.
    
    Рекомендации:
    а. Какие то вопросы или подзадачи могут преследовать цели проверки правильности решения предыдущих задач. Например, может проверяться правильность написания кода, который был написан в предыдущем(-их) пункте(-ов), и тогда следующая за проверкой задача может быть написание нового более правильного кода. (Это был очень частный пример виде дорожный карты)
    б. Рекомендуется несколько раз перепроверить свой ответ, если задача требует этого. А если задача не требует поиска ошибок, например, следует написать текст, то предлагается итеративно улучшать текст, однако, если перед написанием текста следует провести анализ, то конечно его нужно провести в нескольких пунктах
    в. Если задача тяжело резбивается на пункты дорожной карты, то можно оставить общие формулировки, чтобы там на месте разобраться что делать дальше.
Твой ответ (с дорожной картой зи {analysis_depth} пунктов в конце):
""")

        messages = self.context.get_message_history()
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        roadmap_response = self.__call_openai_api(trimmed_messages)

        if roadmap_response is None:
            print("Ошибка: ответ от API не был получен для response_from_chat_GPT_with_chain_of_reasoning.")
            return "Ошибка: не удалось получить ответ от API."

        self.context.add_assistant_message(roadmap_response)

        if debug_reasoning_print:
            print(f"Поставлены следующие задачи:\n{roadmap_response}")

        for iteration in range(analysis_depth):
            self.context.add_user_message(f"Наиподробнейше ответь на вопрос или реши задачу номер {iteration} из дорожной карты для решения вопроса пользователя")

            messages = self.context.get_message_history()
            trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

            _assistant_response = self.__call_openai_api(trimmed_messages)

            if _assistant_response is None:
                print(f"Ошибка: ответ от API не был получен для response_from_chat_GPT_with_chain_of_reasoning при iteration = {iteration}")
                return "Ошибка: не удалось получить ответ от API."

            self.context.add_assistant_message(_assistant_response)

            if debug_reasoning_print:
                print(f"iteration={iteration}\n{_assistant_response}")
            # print(f"Цепочка рассуждений. Для iteration={iteration}:\n{_assistant_response}\n")

        self.context.add_user_message(f"""
На этом мы закончили наши рассуждения! 
Теперь следует, используя всю полученную в ходе исследования информацию дать ответ на вопрос пользователя.

Помни, что пользователь не видел твоих рассуждений, что были внутри блок "режим глубоких рассуждений"
!Мы выходим из режима глубоких рассуждений!

Напиши финальный ответ для пользователя на следующий его вопрос:
{user_message}

Ответ:
""")
        messages = self.context.get_message_history()
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        assistant_response = self.__call_openai_api(trimmed_messages, response_format)

        if assistant_response is None:
            print(f"Ошибка: ответ от API не был получен для response_from_chat_GPT_with_chain_of_reasoning.")
            return "Ошибка: не удалось получить ответ от API."

        if debug_reasoning_print:
            print(f"Финальный ответ:\n{assistant_response}")

        self.context = saved_context
        self.context.add_assistant_message(assistant_response)

        return assistant_response

    def brutal_response_from_chat_GPT(self, user_message: str, images: list = None, response_format: Optional[Type[BaseModel]] = None) -> str:
        """
        Вызывает API с переданным сообщением, не добавляя ничего в основной контекст.

        :param user_message: Сообщение пользователя для отправки в API.
        :param images: Список изображений (если есть).
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :return: Ответ ассистента.
        """
        # Создаем временное сообщение с поддержкой текста и изображений
        temp_message = self.context.brutally_convert_to_message("user", user_message, images)

        # Получаем копию контекста и добавляем временное сообщение
        messages = self.context.get_message_history() + [temp_message]
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        # Вызываем API с временным контекстом
        return self.__call_openai_api(trimmed_messages, response_format)

    @retry(wait=wait_random_exponential(min=1, max=3600),
           stop=stop_after_attempt(10),
           retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    )
    def __call_openai_api(self, messages: list, response_format: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel, None]:
        """
        Вызывает OpenAI API с поддержкой структурированных ответов.

        :param messages: Список сообщений для отправки в API
        :param response_format: Pydantic модель для парсинга ответа
        :return: Ответ в виде строки, Pydantic модели или None при ошибках
        """
        try:
            if response_format:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                    response_format=response_format,
                )

                if response.choices[0].message.refusal:
                    print(f"Отказ модели: {response.choices[0].message.refusal}")
                    return None

                return response.choices[0].message.parsed
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                )

                return response.choices[0].message.content

        except APIError as e:
            print(f"Ошибка API: {e}")
            raise
        except Exception as e:
            print(f"Непредвиденная ошибка: {e}")
            return None

    def __count_tokens_for_single_message(self, message) -> int:
        """
        Подсчитывает количество токенов для одного сообщения.

        :param message: Словарь, представляющий одно сообщение.
        :return: Количество токенов в одном сообщении.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        image_token_count = 2840  # фиксированное количество токенов для изображения

        # Начальное количество токенов на сообщение
        num_tokens = tokens_per_message

        # Подсчет токенов для каждого элемента внутри контента сообщения
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item["type"] == "text":
                    num_tokens += len(encoding.encode(item["text"]))
                elif item["type"] == "image_url":
                    num_tokens += image_token_count
        else:
            # Если контент не является списком, обрабатываем его как обычный текст
            num_tokens += len(encoding.encode(message["content"]))

        # Добавление токенов для имени, если оно существует
        if "name" in message:
            num_tokens += tokens_per_name

        return num_tokens

    def __count_tokens_for_all_messages(self, messages) -> int:
        """
        Подсчитывает общее количество токенов для списка сообщений.

        :param messages: Список сообщений.
        :return: Общее количество токенов для всех сообщений.
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.__count_tokens_for_single_message(message)

        # Добавляем 3 токена для закрытия беседы
        total_tokens += 3
        return total_tokens

    def __trim_context(self, messages: list, max_total_tokens: int) -> list:
        """
        Обрезает контекст до заданного размера в токенах.

        :param messages: Список сообщений.
        :param max_total_tokens: Максимально допустимое количество токенов.
        :return: Обрезанный список сообщений.
        """
        # Подсчитываем токены для каждого сообщения отдельно и получаем общий токен
        token_counts = [self.__count_tokens_for_single_message(message) for message in messages]
        total_tokens = sum(token_counts)

        # Определяем стартовый индекс в зависимости от типа первого сообщения
        start_index = 1 if messages[0]["role"] == "system" else 0

        # Удаление дублирующихся системных сообщений
        while total_tokens > max_total_tokens and len(messages) > start_index + 1:
            for i in range(start_index, len(messages) - 1):
                if messages[i]["role"] == "system":
                    # Проверяем, есть ли точно такое же системное сообщение позже
                    duplicate_found = any(
                        msg["role"] == "system" and msg["content"] == messages[i]["content"]
                        for msg in messages[i + 1:]
                    )
                    if duplicate_found:
                        # Удаляем первое из повторяющихся системных сообщений
                        total_tokens -= token_counts[i]  # Вычитаем токены удаленного сообщения
                        del messages[i]
                        del token_counts[i]  # Удаляем соответствующее значение из списка токенов
                        break
            else:
                # Если не осталось повторяющихся системных сообщений, прерываем цикл
                break

        # Удаление старых сообщений, если токены все еще превышают лимит
        while total_tokens > max_total_tokens and len(messages) > start_index + 1:
            total_tokens -= token_counts[start_index]  # Вычитаем токены удаленного сообщения
            del messages[start_index]
            del token_counts[start_index]

        if total_tokens > max_total_tokens:
            print("Предупреждение: Контекст не может быть уменьшен до заданного размера.")

            # Удаление уникальных системных сообщений с конца
            for i in range(len(messages) - 1, start_index - 1, -1):
                if messages[i]["role"] == "system":
                    total_tokens -= token_counts[i]  # Вычитаем токены удаленного сообщения
                    del messages[i]
                    del token_counts[i]
                    if total_tokens <= max_total_tokens:
                        break

        return messages

    def clone(self):
        """
        Создает глубокую копию текущего объекта ChatGPTAgent, включая все параметры и контекст сообщений.

        :return: Новый экземпляр ChatGPTAgent с идентичными параметрами и контекстом.
        """
        # Создаем новый объект ChatGPTAgent с теми же параметрами
        cloned_agent = ChatGPTAgent(
            api_key=self.api_key,
            organization=self.client.organization,
            model_name=self.model_name,
            mode=self.context.mode,
            task_prompt=self.context.task_prompt,
            max_total_tokens=self.max_total_tokens,
            max_response_tokens=self.max_response_tokens,
            temperature=self.temperature
        )

        # Копируем контекст сообщений, используя метод clone из MessageContext
        cloned_agent.context = self.context.clone()

        return cloned_agent
