from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
from typing import Optional, Union, Type, List, Dict, Any
import tiktoken
import os
import base64
import requests
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import json
import re
import time
import sys
import logging
import traceback

from src.debug_tracer import DebugTracer
from src.utils import load_prompts
from src.messages_meta_data_manager import MessagesWithMetaData
from src.message_manager import MessageContext


class DeepSeekRouterError(Exception):
    """Пользовательский класс ошибки для обработки пустых ответов от OpenRouter API"""
    pass


class ChatLLMAgent:
    """
    Класс ChatLLMAgent взаимодействует с API LLM, используя MessageContext для управления контекстом сообщений.
    """

    def __init__(self, model_name: str, mode: int, task_prompt: str = None,
                 openai_api_key: str = None, openai_organization: str = None,
                 openrouter_api_key: str = None, use_openai_or_openrouter: str = None,
                 max_total_tokens: int = 32000, max_response_tokens: int = 4095, temperature: float = 0.0):
        """
        Инициализирует агент с API-ключом и настраивает контекст сообщений.

        Режимы (моды):
            1. Очищает контекст перед каждым добавлением нового пользовательского сообщения. В пустой контекст добавляется task_prompt и сообщение пользователя.
            2. Сохраняет контекст между запросами. task_prompt добавляется только один раз в самое начало, после чего каждое новое сообщение добавляется поверх предыдущих. При изменении task_prompt добавляется новое системное сообщение.
            3. Сохраняет контекст между запросами. task_prompt добавляется перед каждым пользовательским сообщением, но не перед сообщениями ассистента.

        :param model_name: Название модели для API OpenAI или OpenRouter.
        :param mode: Режим работы контекста.
        :param task_prompt: Задание, добавляемое в контекст.
        :param openai_api_key: API-ключ для доступа к OpenAi.
        :param openai_organization: Organization ID для доступа к OpenAi.
        :param openrouter_api_key: API-ключ для доступа к OpenRouter.
        :param use_openai_or_openrouter: Провайдер который используется: либо 'openai', либо 'openrouter'
        :param max_total_tokens: Максимальное количество токенов для запроса.
        :param max_response_tokens: Максимальное количество токенов в ответе.
        :param temperature: Параметр температуры для API (креативность ответов).
        """
        if use_openai_or_openrouter != "openai" and use_openai_or_openrouter != "openrouter":
            raise ValueError(f"Выбран неизвестный провайдер {use_openai_or_openrouter}. Выберите либо 'openai', либо 'openrouter'.")
        if use_openai_or_openrouter == "openai":
            if model_name.count("/"):
                raise ValueError(f"Нам кажется, что вы указали название модели для openrouter, хотя указали, что используете openai."
                                 f"model_name={model_name}, use_openai_or_openrouter={use_openai_or_openrouter}.")
            elif not openai_api_key or not openai_organization:
                raise ValueError(f"Вы используете провайдера openai, но не указали openai_api_key и openai_organization.")
        else:  # если openrouter
            if not model_name.count("/"):
                raise ValueError(f"Нам кажется, что вы указали название модели для openai, хотя указали, что используете openrouter."
                                 f"model_name={model_name}, use_openai_or_openrouter={use_openai_or_openrouter}")
            elif not openrouter_api_key:
                raise ValueError(f"Вы используете провайдера openrouter, но не указали openrouter_api_key.")

        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.openai_organization = openai_organization
        self.openrouter_api_key = openrouter_api_key
        self.use_openai_or_openrouter = use_openai_or_openrouter
        self.max_total_tokens = max_total_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

        if use_openai_or_openrouter == "openai":
            self.client = OpenAI(
                organization=openai_organization,
                api_key=openai_api_key
            )

        self.context = MessageContext(mode=mode, task_prompt=task_prompt)
        self.messages_meta_data: MessagesWithMetaData = MessagesWithMetaData(self.context.messages)
        self.max_llm_calling_count: int = sys.maxsize

        if use_openai_or_openrouter == "openai":
            self.call_llm = self.__call_openai_api
        else:  # если openrouter
            self.call_llm = self.__call_open_router_api

        self.tracer = None

    def initialize_context_optimization(self, debug_reasoning_print: bool = False):
        """
        Инициализирует функции оптимизации контекста в MessagesWithMetaData.

        :param debug_reasoning_print: Флаг для вывода отладочной информации
        """
        # Предотвращаем повторную инициализацию
        if hasattr(self.messages_meta_data.__class__, 'safe_replace_prompt'):
            if debug_reasoning_print:
                print("Методы оптимизации контекста уже инициализированы")
            return

        # try:
        #     # Добавляем методы в класс MessagesWithMetaData
        #     MessagesWithMetaDataClass = self.messages_meta_data.__class__
        #
        #     # Определяем и добавляем методы
        #     setattr(MessagesWithMetaDataClass, '_is_prompt_already_shortened', _is_prompt_already_shortened)
        #     setattr(MessagesWithMetaDataClass, '_mark_prompt_as_shortened', _mark_prompt_as_shortened)
        #     setattr(MessagesWithMetaDataClass, 'find_messages_by_type', find_messages_by_type)
        #     setattr(MessagesWithMetaDataClass, 'replace_prompt_in_message', replace_prompt_in_message)
        #     setattr(MessagesWithMetaDataClass, 'replace_prompts_by_type', replace_prompts_by_type)
        #     setattr(MessagesWithMetaDataClass, 'safe_replace_prompt', safe_replace_prompt)
        #
        #     if debug_reasoning_print:
        #         print("Методы оптимизации контекста успешно инициализированы")
        # except Exception as e:
        #     error_traceback = traceback.format_exc()
        #     logging.error(f"Ошибка при инициализации методов оптимизации контекста: {str(e)}\n{error_traceback}")
        #     if debug_reasoning_print:
        #         print(f"Ошибка при инициализации методов оптимизации контекста: {str(e)}")

    def response_from_LLM(self, user_message: str, images: list = None,
                          response_format: Optional[Type[BaseModel]] = None,
                          model_name: str = None) -> str:
        """
        Добавляет сообщение пользователя, получает ответ от чата LLM и добавляет его в контекст.

        :param user_message: Сообщение пользователя для добавления в контекст и отправки в API.
        :param images: Список изображений (если есть).
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :param model_name: по умолчанию используется модель указанная при инициализации ChatLLMAgent,
                но через эту переменную вы можете указать другую модель
        :return: Ответ ассистента.
        """
        self.context.add_user_message(user_message, images)

        messages = self.context.get_message_history()
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        assistant_response = self.call_llm(
            messages=trimmed_messages, response_format=response_format, model_name=model_name)

        if assistant_response is None:
            print("Ошибка: ответ от API не был получен для response_from_LLM.")
            return "Ошибка: не удалось получить ответ от API."

        self.context.add_assistant_message(assistant_response)
        return assistant_response

    def response_from_LLM_with_decomposition(self, analysis_depth: int, user_message: str,
                                             images: list = None,
                                             preserve_user_messages_post_analysis: bool = True,
                                             response_format: Optional[Type[BaseModel]] = None,
                                             debug_reasoning_print=False,
                                             model_name: str = None) -> str:
        """
        Делает то же самое, что и response_from_LLM, но с использованием метода цепочки рассуждений.
        Она позволяет глубже анализировать запросы, формируя структурированный и обоснованный ответ.

        :param analysis_depth: Определяет количество итераций для анализа запроса. Чем больше значение, тем более детальный анализ будет проведен.
        :param user_message: Сообщение от пользователя, которое требует анализа и ответа.
        :param images: Список изображений, связанных с пользовательским сообщением (список путей до изображений). По умолчанию None.
        :param preserve_user_messages_post_analysis: Указывает, следует ли сохранять входящее пользовательское сообщения после анализа. По умолчанию True.
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :param debug_reasoning_print: Вывод отладочной информации в консоль.
        :param model_name: по умолчанию используется модель указанная при инициализации ChatLLMAgent,
                но через эту переменную вы можете указать другую модель.
        :return: Ответ ассистента.
        """
        if analysis_depth < 1:
            raise Exception(
                "Ошибка: глубина анализа должна быть не менее 1. Пожалуйста, укажите корректное значение для analysis_depth.")

        copied_context = self.context.clone()
        if preserve_user_messages_post_analysis:
            self.context.add_user_message(user_message, images)

        if debug_reasoning_print:
            print(
                f"Запуск response_from_LLM_with_decomposition\nВходное сообщение:\n{user_message}\nВходные изображения: {images}")

        saved_context = self.context
        self.context = copied_context

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

        roadmap_response = self.call_llm(messages=trimmed_messages, model_name=model_name)

        if roadmap_response is None:
            print("Ошибка: ответ от API не был получен для response_from_LLM_with_decomposition.")
            return "Ошибка: не удалось получить ответ от API."

        self.context.add_assistant_message(roadmap_response)

        if debug_reasoning_print:
            print(f"Поставлены следующие задачи:\n{roadmap_response}")

        for iteration in range(analysis_depth):
            self.context.add_user_message(
                f"Наиподробнейше ответь на вопрос или реши задачу номер {iteration} из дорожной карты для решения вопроса пользователя")

            messages = self.context.get_message_history()
            trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

            _assistant_response = self.call_llm(messages=trimmed_messages, model_name=model_name)

            if _assistant_response is None:
                print(
                    f"Ошибка: ответ от API не был получен для response_from_LLM_with_decomposition при iteration = {iteration}")
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

        assistant_response = self.call_llm(messages=trimmed_messages, response_format=response_format, model_name=model_name)

        if assistant_response is None:
            print(f"Ошибка: ответ от API не был получен для response_from_LLM_with_decomposition.")
            return "Ошибка: не удалось получить ответ от API."

        if debug_reasoning_print:
            print(f"Финальный ответ:\n{assistant_response}")

        self.context = saved_context
        self.context.add_assistant_message(assistant_response)

        return assistant_response

    def brutal_response_from_LLM(self, user_message: str, images: list = None,
                                      response_format: Optional[Type[BaseModel]] = None,
                                      model_name: str = None) -> str:
        """
        Вызывает API с переданным сообщением, не добавляя ничего в основной контекст.

        :param user_message: Сообщение пользователя для отправки в API.
        :param images: Список изображений (если есть).
        :param response_format: Pydantic модель для парсинга ответа (если требуется).
        :param model_name: по умолчанию используется модель указанная при инициализации ChatLLMAgent,
                но через эту переменную вы можете указать другую модель
        :return: Ответ ассистента.
        """
        # Создаем временное сообщение с поддержкой текста и изображений
        temp_message = self.context.brutally_convert_to_message("user", user_message, images)

        # Получаем копию контекста и добавляем временное сообщение
        messages = self.context.get_message_history() + [temp_message]
        trimmed_messages = self.__trim_context(messages, self.max_total_tokens - self.max_response_tokens)

        # Вызываем API с временным контекстом
        return self.call_llm(messages=trimmed_messages, response_format=response_format, model_name=model_name)

    @retry(wait=wait_random_exponential(min=1, max=3600),
           stop=stop_after_attempt(10),
           retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
           )
    def __call_openai_api(self, messages: List[Dict[str, Any]], response_format: Optional[Type[BaseModel]] = None,
                          model_name: str = None) -> Union[str, BaseModel, None]:
        """
        Вызывает OpenAI API с поддержкой структурированных ответов.

        :param messages: Список сообщений для отправки в API
        :param response_format: Pydantic модель для парсинга ответа
        :param model_name: по умолчанию используется модель указанная при инициализации ChatLLMAgent,
                но через эту переменную вы можете указать другую модель
        :return: Ответ в виде строки, Pydantic модели или None при ошибках
        """
        if not model_name:
            model_name = self.model_name

        if self.use_openai_or_openrouter == "openai" and model_name and model_name.count("/"):
            raise ValueError(f"Нам кажется, что вы указали название модели для openrouter, хотя указали, что используете openai."
                             f"model_name={model_name}, use_openai_or_openrouter={self.use_openai_or_openrouter}.")
        elif self.use_openai_or_openrouter == "openrouter" and model_name and not model_name.count("/"):
            raise ValueError(f"Нам кажется, что вы указали название модели для openai, хотя указали, что используете openrouter."
                             f"model_name={model_name}, use_openai_or_openrouter={self.use_openai_or_openrouter}")

        try:
            if response_format:
                response = self.client.beta.chat.completions.parse(
                    model=model_name,
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
            # print("Warning: OpenAi tokenaizer not found. Using o200k_base encoding.")
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
        # Сохраняем оригинальный список сообщений для логирования
        original_messages = messages.copy()

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

        # Логирование обрезанных сообщений, если трассировщик доступен
        if hasattr(self, 'tracer') and self.tracer:
            self.tracer.log_trimmed_messages(original_messages, messages)

        return messages

    # Внимание: в данной реализации не копирует tracer
    def clone(self):
        """
        Создает полную глубокую копию текущего агента со всеми компонентами.

        :return: Новый экземпляр ChatLLMAgent
        """
        cloned_agent = ChatLLMAgent(
            model_name=self.model_name,
            mode=self.context.mode,
            task_prompt=self.context.task_prompt,
            openai_api_key=self.openai_api_key,
            openai_organization=self.openai_organization,
            openrouter_api_key=self.openrouter_api_key,
            use_openai_or_openrouter=self.use_openai_or_openrouter,
            max_total_tokens=self.max_total_tokens,
            max_response_tokens=self.max_response_tokens,
            temperature=self.temperature
        )

        cloned_agent.context = self.context.clone()

        cloned_agent.messages_meta_data = self.messages_meta_data.clone(cloned_agent.context.messages)

        cloned_agent.max_llm_calling_count = self.max_llm_calling_count

        if hasattr(self.messages_meta_data.__class__, 'safe_replace_prompt'):
            cloned_agent.initialize_context_optimization(False)

        if self.use_openai_or_openrouter == "openai" and cloned_agent.openai_api_key:
            cloned_agent.client = OpenAI(
                organization=cloned_agent.openai_organization,
                api_key=cloned_agent.openai_api_key
            )

        return cloned_agent

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((APIError, APIConnectionError, APITimeoutError, RateLimitError, DeepSeekRouterError)),
        reraise=True
    )
    def __call_open_router_api(self, messages: List[Dict[str, Any]],
                               response_format: Optional[Type[BaseModel]] = None,
                               model_name: str = None) -> Union[str, BaseModel, None]:
        """
        Вызывает OpenRouter API с обработкой пустых ответов и автоматическими повторами.
        """
        if not model_name:
            model_name = self.model_name

        # Валидация настроек модели и провайдера
        if self.use_openai_or_openrouter == "openai" and model_name and model_name.count("/"):
            raise ValueError(f"Нам кажется, что вы указали название модели для openrouter, хотя указали, что используете openai."
                             f"model_name={model_name}, use_openai_or_openrouter={self.use_openai_or_openrouter}.")
        elif self.use_openai_or_openrouter == "openrouter" and model_name and not model_name.count("/"):
            raise ValueError(f"Нам кажется, что вы указали название модели для openai, хотя указали, что используете openrouter."
                             f"model_name={model_name}, use_openai_or_openrouter={self.use_openai_or_openrouter}")

        print("🤖 Инициирование запроса к OpenRouter API")

        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")

        client = None
        try:
            # Создаем клиента с отключенным внутренним retry механизмом
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                max_retries=0
            )

            # Подготовка сообщений для запроса
            converted_messages = self._convert_and_validate_messages(messages)

            # Формирование параметров запроса
            request_parameters = {
                "model": model_name,
                "messages": converted_messages,
                "max_tokens": self.max_response_tokens,
                "temperature": self.temperature,
                "extra_headers": {
                    "HTTP-Referer": "https://your-site.com",
                    "X-Title": "Your Application Name"
                },
                # Правильный способ передачи параметров маршрутизации - через extra_body
                "extra_body": {
                    "provider": {
                        "sort": "throughput"  # Приоритезируем стабильность над ценой
                    }
                }
            }

            # Добавление параметров формата ответа, если требуется
            if response_format is not None:
                request_parameters["response_format"] = {"type": "json_object"}

            # Выполняем запрос
            api_response = client.chat.completions.create(**request_parameters)

            # Проверка базовой структуры ответа
            if not api_response or not hasattr(api_response, 'choices') or not api_response.choices:
                raise DeepSeekRouterError("Пустой ответ от API (отсутствуют choices)")

            # Получение контента из ответа
            content = api_response.choices[0].message.content
            provider = getattr(api_response, 'provider', 'unknown')

            # КЛЮЧЕВАЯ ПРОВЕРКА: контент не должен быть пустым
            if content is None or (isinstance(content, str) and content.strip() == ""):
                # Собираем диагностическую информацию
                usage_info = ""
                if hasattr(api_response, 'usage'):
                    usage_info = f", tokens: {api_response.usage.completion_tokens}/{api_response.usage.prompt_tokens}"

                # Формируем информативное сообщение об ошибке
                error_msg = f"Получен пустой ответ от провайдера {provider}{usage_info}"
                print(f"⚠️ {error_msg}")

                # Вызываем исключение для активации повторной попытки
                raise DeepSeekRouterError(error_msg)

            # Логирование успешного ответа
            # print(f"✅ Получен ответ от {provider}")

            # Обработка формата ответа, если требуется
            if response_format is not None:
                try:
                    return response_format.parse_raw(content)
                except Exception as e:
                    raise ValueError(f"Ошибка парсинга модели: {str(e)}") from e

            return content

        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as api_error:
            # Формируем информативное сообщение об ошибке
            error_details = []

            if hasattr(api_error, 'message'):
                error_details.append(f"сообщение: {api_error.message}")
            if hasattr(api_error, 'status_code'):
                error_details.append(f"HTTP статус: {api_error.status_code}")
            if hasattr(api_error, 'code'):
                error_details.append(f"код ошибки: {api_error.code}")

            error_info = ', '.join(error_details) if error_details else 'детали отсутствуют'
            error_message = f"[Ошибка API] {type(api_error).__name__}: {error_info}"

            # Логирование ошибки с деталями
            print(f"❌ {error_message}")

            # Пробрасываем исключение для механизма retry
            raise

        except DeepSeekRouterError as custom_error:
            # Просто пробрасываем наши собственные ошибки
            raise

        except Exception as general_error:
            # Обработка непредвиденных ошибок
            print(f"🔥 Непредвиденная ошибка: {str(general_error)}")
            raise

        finally:
            # Гарантированное освобождение ресурсов
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _convert_and_validate_messages(self, messages: list) -> list:
        """
        Конвертация и валидация сообщений для DeepSeek

        :param messages: Список сообщений для конвертации
        :return: Список сконвертированных сообщений
        """
        converted = []

        for idx, msg in enumerate(messages):
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Некорректное сообщение в позиции {idx}")

            new_msg = {
                "role": msg["role"],
                "content": self._process_content(msg["content"])
            }

            converted.append(new_msg)

        return converted

    def _process_content(self, content) -> list:
        """
        Обработка контента сообщения с валидацией

        :param content: Содержимое сообщения
        :return: Обработанный контент
        """
        processed = []

        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        for item in content:
            if item["type"] == "text":
                processed.append(item)
            elif item["type"] == "image_url":
                processed.append(self._process_image(item["image_url"]["url"]))
            else:
                raise ValueError(f"Неподдерживаемый тип контента: {item['type']}")

        return processed

    def _process_image(self, image_url: str) -> dict:
        """
        Обработка и валидация изображений с улучшенной логикой для локальных файлов.

        :param image_url: URL изображения или путь к файлу
        :return: Обработанное представление изображения
        """
        # Сначала проверяем, выглядит ли это как локальный путь
        if self._is_local_path(image_url):
            # Проверяем существование файла
            if not self._is_local_file_exists(image_url):
                raise FileNotFoundError(
                    f"Локальный файл изображения не найден или недоступен: {image_url}. "
                    f"Убедитесь, что файл существует и у вас есть права на его чтение."
                )

            # Конвертируем локальный файл в base64
            try:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": self._local_image_to_base64(image_url),
                        "detail": "auto"
                    }
                }
            except Exception as e:
                raise ValueError(f"Ошибка при конвертации локального изображения в base64: {str(e)}")

        # Если это не локальный путь, проверяем как URL
        if not self._is_valid_image_url(image_url):
            raise ValueError(
                f"Недопустимый URL изображения: {image_url}. "
                f"URL должен начинаться с http:// или https:// и указывать на изображение."
            )

        return {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "auto"
            }
        }

    def _is_local_path(self, path: str) -> bool:
        """
        Проверяет, выглядит ли строка как локальный путь к файлу.

        :param path: Строка для проверки
        :return: True, если строка выглядит как локальный путь
        """
        # Проверяем различные форматы локальных путей
        local_path_indicators = [
            '/',           # Unix абсолютные пути
            './',          # Относительные пути
            '../',         # Родительские директории
            '~/',          # Домашняя директория
        ]

        # Windows пути
        if len(path) >= 3 and path[1:3] == ':\\':  # C:\, D:\, etc.
            return True

        # Unix/Linux/Mac пути
        for indicator in local_path_indicators:
            if path.startswith(indicator):
                return True

        # Проверяем, содержит ли путь специфичные для файловой системы символы
        # но не содержит схемы протокола
        if not path.startswith(('http://', 'https://', 'ftp://', 'data:')):
            # Если содержит символы пути и точку (расширение файла)
            if ('/' in path or '\\' in path) and '.' in path:
                return True

        return False

    def _is_local_file_exists(self, path: str) -> bool:
        """
        Проверяет, существует ли локальный файл.

        :param path: Путь к файлу
        :return: True, если файл существует и доступен
        """
        try:
            return os.path.isfile(path) and os.access(path, os.R_OK)
        except (OSError, TypeError):
            return False

    def _local_image_to_base64(self, file_path: str) -> str:
        """
        Конвертация локального файла в base64 с улучшенной обработкой ошибок.

        :param file_path: Путь к файлу изображения
        :return: Строка в формате Data URL (data:mime/type;base64,...)
        """
        try:
            # Проверяем размер файла перед чтением
            file_size = os.path.getsize(file_path)
            max_size = 20 * 1024 * 1024  # 20MB лимит

            if file_size > max_size:
                raise ValueError(f"Размер файла ({file_size} bytes) превышает максимально допустимый ({max_size} bytes)")

            with open(file_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = self._get_mime_type(file_path)
                return f"data:{mime_type};base64,{encoded}"

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл изображения не найден: {file_path}")
        except PermissionError:
            raise PermissionError(f"Нет прав доступа к файлу: {file_path}")
        except OSError as e:
            raise OSError(f"Ошибка при чтении файла {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Ошибка кодирования изображения {file_path}: {str(e)}")

    def _get_mime_type(self, file_path: str) -> str:
        """
        Определяет MIME-тип файла по расширению с улучшенной поддержкой форматов.

        :param file_path: Путь к файлу
        :return: MIME-тип файла
        """
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.svg': 'image/svg+xml'
        }

        mime_type = mime_types.get(ext)
        if not mime_type:
            raise ValueError(
                f"Неподдерживаемый формат изображения: {ext}. "
                f"Поддерживаемые форматы: {', '.join(mime_types.keys())}"
            )

        return mime_type

    def _is_valid_image_url(self, url: str) -> bool:
        """
        Валидация URL изображения с поддержкой data URLs и обычных HTTP URLs.

        :param url: URL для проверки (может быть http/https URL или data URL)
        :return: True, если URL валиден
        """
        try:
            # Проверяем data URLs (data:image/...;base64,...)
            if url.startswith('data:'):
                # Проверяем формат data URL для изображений
                if url.startswith('data:image/'):
                    # Проверяем наличие base64 части
                    if ';base64,' in url:
                        # Получаем MIME тип
                        mime_part = url.split(';')[0].replace('data:', '')
                        # Список поддерживаемых MIME типов
                        valid_mime_types = [
                            'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
                            'image/webp', 'image/bmp', 'image/tiff', 'image/svg+xml'
                        ]
                        return mime_part in valid_mime_types
                return False

            # Проверяем обычные HTTP/HTTPS URLs
            if not url.startswith(('http://', 'https://')):
                return False

            # Выполняем HEAD запрос с таймаутом для HTTP URLs
            resp = requests.head(url, timeout=10, allow_redirects=True)

            # Проверяем статус код
            if resp.status_code != 200:
                return False

            # Проверяем Content-Type
            content_type = resp.headers.get('Content-Type', '').lower()
            return content_type.startswith('image/')

        except requests.RequestException:
            # Любые сетевые ошибки считаем невалидным URL
            return False
        except Exception:
            # Любые другие ошибки тоже считаем невалидным URL
            return False

    def response_from_LLM_with_hierarchical_recursive_decomposition(
        self,
        user_message: str,
        images: list = None,
        model_name: str = None,
        larger_model_name: str = None,
        max_llm_calling_count: int = 1000,
        preserve_user_messages_post_analysis: bool = True,
        response_format: Optional[Type["BaseModel"]] = None,
        debug_reasoning_print: bool = False,
    ) -> str:
        """
        Главная функция, реализующая рекурсивную схему решения задачи
        с разбиением на подзадачи. Включает логику:
          1) Формулировка задачи и критериев
          2) Попытка решения (start_solution_gen_prompt)
          3) Проверка решения (solution_verification_prompt)
          4) Определение действия (action_manager_prompt -> code 'а' | 'б' | 'в' | 'г')
          5) При необходимости декомпозиция на подзадачи (decompose_task_prompt),
             решение каждого из них рекурсивно
          6) Возврат итогового решения (final_solution_text_generator_prompt),
             когда условие "а" достигнуто.
        Также мы используем 2 llm модели для разных задач.
        Считается,что larger_model_name более мощная, чем model_name.

        :param user_message: Текст задачи от пользователя
        :param images: Список изображений (если есть)
        :param model_name: используется для задач генерации теоретического материала,
                при определение следующего действия, при формировании текста обобщающего решённые подзадачи
                но по умолчанию используется модель указанная при инициализации ChatLLMAgent,
        :param larger_model_name: используется для задач создания критериев качества,
                при генерации самого решения, при проверки решения,
                при декомпозиции задачи и при генерации текста финального решения
                но по умолчанию используется модель указанная при инициализации ChatLLMAgent,
        :param max_llm_calling_count: Максимальное количество вызовов LLM
        :param preserve_user_messages_post_analysis: Сохранять ли сообщение пользователя после анализа
        :param response_format: Формат ответа (если требуется)
        :param debug_reasoning_print: Флаг для вывода отладочной информации
        :return: Итоговое решение задачи
        """
        self.tracer = DebugTracer(messages_meta_data=self.messages_meta_data) if debug_reasoning_print else None
        tracer = self.tracer

        # Инициализация методов оптимизации контекста
        self.initialize_context_optimization(debug_reasoning_print)

        # Загрузка промптов: полных и сокращенных версий
        full_prompts, shortened_prompts = load_prompts()

        # Оригинальные промпты
        main_recursive_decomposition_prompt = full_prompts.get("main_recursive_decomposition_prompt", "")
        task_statement_prompt = full_prompts.get("task_statement_prompt", "")
        theory_gen_prompt = full_prompts.get("theory_gen_prompt", "")
        quality_assessment_criteria_prompt = full_prompts.get("quality_assessment_criteria_prompt", "")
        start_solution_gen_prompt = full_prompts.get("start_solution_gen_prompt", "")
        solution_verification_prompt = full_prompts.get("solution_verification_prompt", "")
        action_manager_prompt = full_prompts.get("action_manager_prompt", "")
        final_solution_text_generator_prompt = full_prompts.get("final_solution_text_generator_prompt", "")
        re_solve_unsuccessful_decision = full_prompts.get("re_solve_unsuccessful_decision", "")
        continue_solution_prompt = full_prompts.get("continue_solution_prompt", "")
        decompose_task_prompt = full_prompts.get("decompose_task_prompt", "")
        finish_task_after_solving_subtasks_prompt = full_prompts.get("finish_task_after_solving_subtasks_prompt", "")

        # Сокращенные версии промптов
        theory_gen_shortened_prompt = shortened_prompts.get("theory_gen_prompt", "")
        quality_assessment_criteria_shortened_prompt = shortened_prompts.get("quality_assessment_criteria_prompt", "")
        start_solution_gen_shortened_prompt = shortened_prompts.get("start_solution_gen_prompt", "")
        solution_verification_shortened_prompt = shortened_prompts.get("solution_verification_prompt", "")
        action_manager_shortened_prompt = shortened_prompts.get("action_manager_prompt", "")
        re_solve_unsuccessful_decision_shortened_prompt = shortened_prompts.get("re_solve_unsuccessful_decision", "")
        continue_solution_shortened_prompt = shortened_prompts.get("continue_solution_prompt", "")
        decompose_task_shortened_prompt = shortened_prompts.get("decompose_task_prompt", "")
        finish_task_after_solving_subtasks_shortened_prompt = shortened_prompts.get("finish_task_after_solving_subtasks_prompt", "")

        # --------------------------------------------------------------------------
        # Вспомогательные функции для работы с плейсхолдерами в промптах
        # --------------------------------------------------------------------------
        def get_prompt_placeholders(current_level, current_task_id):
            """
            Формирует словарь с заменами для плейсхолдеров в промптах
            на основе текущего уровня и ID задачи

            :param current_level: Текущий уровень задачи
            :param current_task_id: Идентификатор текущей задачи
            :return: Словарь с заменами плейсхолдеров
            """
            placeholders = {}

            # Очищаем current_task_id от конечной точки для использования в parent_id
            clean_task_id = current_task_id.rstrip(".")

            # Формируем идентификатор родительской задачи, если мы не на исходном уровне
            parent_id = ".".join(clean_task_id.split(".")[:-1]) if current_level > 0 else ""
            # Добавляем точку к parent_id, если он не пустой
            if parent_id:
                parent_id += "."

            # Базовые плейсхолдеры
            placeholders["task_id"] = current_task_id

            # Индикаторы уровня и тип задачи
            if current_level > 0:
                placeholders["level_indicator"] = " (УРОВЕНЬ ПОДЗАДАЧИ)"
                placeholders["task_context"] = "подзадачей основной проблемы"
                placeholders["subtask_indicator"] = "подзадачей"
                placeholders["parent_reference"] = f"Учти, что эта задача является частью задачи {parent_id} и должна согласовываться с общим подходом к решению."
            else:
                placeholders["level_indicator"] = ""
                placeholders["task_context"] = "основной задачей"
                placeholders["subtask_indicator"] = ""
                placeholders["parent_reference"] = ""

            if current_level > 0:
                placeholders["hierarchy_reminder"] = f"Данная задача является подзадачей {current_task_id} в иерархии рекурсивной декомпозиции."
                placeholders["context_reminder"] = f"Помни, что это решение должно быть интегрировано в контекст родительской задачи {parent_id} Убедись, что твой подход согласуется с общей стратегией решения."  # Убрана точка после parent_id
            else:
                placeholders["hierarchy_reminder"] = "Это основная задача в алгоритме рекурсивной декомпозиции."
                placeholders["context_reminder"] = "Поскольку это основная задача, твое решение должно быть полным и удовлетворять всем установленным критериям качества."

            return placeholders

        def localize_prompt(prompt, placeholders):
            """
            Заменяет плейсхолдеры в промпте на их значения

            :param prompt: Исходный текст промпта
            :param placeholders: Словарь с заменами плейсхолдеров
            :return: Локализованный текст промпта
            """
            localized_prompt = prompt
            for key, value in placeholders.items():
                localized_prompt = localized_prompt.replace(f"{{{key}}}", value)
            return localized_prompt

        # --------------------------------------------------------------------------
        # Вспомогательные функции (внутренние). При желании можно их вынести наружу.
        # --------------------------------------------------------------------------
        def parsing_action_function(answer: str, debug_print: bool = False) -> str:
            """
            Извлекает код действия из ответа LLM, поддерживая различные форматы.

            :param answer: Текст ответа от LLM
            :param debug_print: Флаг для вывода отладочной информации
            :return: Строка с кодом действия ('а', 'б', 'в', 'г')
            """
            if debug_print:
                print(f"Парсинг действия из ответа: {answer[:100]}...")

            # Способ 1: Пытаемся найти и разобрать JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', answer, re.DOTALL)
            if not json_match:
                # Ищем JSON без обрамления кодовыми блоками
                json_match = re.search(r'(\{[^{]*"action"[^}]*\})', answer, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)

                    # Проверяем наличие поля action
                    if "action" in data:
                        action = data["action"].lower()

                        # Проверяем латиницу и кириллицу
                        if action in ['a', 'а']:
                            return 'а'  # Решение удовлетворяет критериям
                        elif action in ['b', 'б']:
                            return 'б'  # Легкие ошибки, пересобрать
                        elif action in ['c', 'в']:
                            return 'в'  # Решение неполное, продолжить
                        elif action in ['d', 'г']:
                            return 'г'  # Серьезные ошибки, декомпозиция

                        if debug_print:
                            print(f"Найдено нестандартное значение action в JSON: {action}")
                except json.JSONDecodeError as e:
                    if debug_print:
                        print(f"Ошибка разбора JSON: {e}")

            # Способ 2: Ищем код действия в тексте
            # Сначала ищем кириллические буквы с соответствующим контекстом
            action_patterns = [
                r'(действие|action|вариант|выбор|решение)[^а-яА-Я]*([абвг])\)',
                r'([абвг])\s*\)',
                r'"action"\s*:\s*"([абвг])"',
                r'действие\s*([абвг])',
                r'выбираю\s*([абвг])'
            ]

            for pattern in action_patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    letter_group = 1 if pattern == r'([абвг])\s*\)' else 2
                    letter = match.group(letter_group).lower()
                    if debug_print:
                        print(f"Найден код действия по шаблону: {letter}")
                    return letter

            # Просто ищем кириллические буквы
            cyrillic_match = re.search(r'[абвг]', answer, re.IGNORECASE)
            if cyrillic_match:
                letter = cyrillic_match.group(0).lower()
                if debug_print:
                    print(f"Найдена кириллическая буква: {letter}")
                return letter

            # Ищем латинские буквы и преобразуем в кириллические
            latin_match = re.search(r'[abcd]', answer, re.IGNORECASE)
            if latin_match:
                latin_letter = latin_match.group(0).lower()
                # Таблица соответствия
                conversion = {'a': 'а', 'b': 'б', 'c': 'в', 'd': 'г'}
                if debug_print:
                    print(f"Найдена латинская буква: {latin_letter}, преобразуется в {conversion[latin_letter]}")
                return conversion[latin_letter]

            # Ищем ключевые слова для определения действия
            keywords = {
                'удовлетворяет': 'а', 'завершено': 'а', 'готово': 'а', 'успешно': 'а', 'соответствует': 'а',
                'исправить': 'б', 'легк': 'б', 'пересобрать': 'б', 'незначительн': 'б',
                'продолжить': 'в', 'неполн': 'в', 'дополнить': 'в', 'доработать': 'в',
                'серьезн': 'г', 'сложн': 'г', 'декомпозиц': 'г', 'разбить': 'г', 'подзадач': 'г'
            }

            for keyword, action in keywords.items():
                if keyword in answer.lower():
                    if debug_print:
                        print(f"Найдено ключевое слово: {keyword}, соответствует действию {action}")
                    return action

            # По умолчанию возвращаем 'г' (декомпозиция)
            if debug_print:
                print("Не удалось определить действие, возвращаем 'г' по умолчанию")

            return 'г'

        def parsing_decompose_task_function(answer: str, debug_print: bool = False) -> List[str]:
            """
            Извлекает подзадачи из ответа LLM, поддерживая как JSON, так и нумерованные списки.

            :param answer: Текст ответа от LLM
            :param debug_print: Флаг для вывода отладочной информации
            :return: Список строк с описаниями подзадач
            """
            if debug_print:
                print(f"Оригинальный ответ: {answer[:200]}...")

            # Способ 1: Пытаемся найти и разобрать JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)

                    # Проверяем структуру JSON
                    if "subtasks" in data and isinstance(data["subtasks"], list):
                        subtasks = []

                        for task in data["subtasks"]:
                            if isinstance(task, dict):
                                # Формируем строку описания подзадачи из полей JSON
                                title = task.get('title', '')
                                goal = task.get('goal', '')

                                # Если есть и заголовок, и цель, объединяем их
                                if title and goal:
                                    task_str = f"{title}: {goal}"
                                else:
                                    task_str = title or goal

                                if task_str:
                                    subtasks.append(task_str)

                        if subtasks and debug_print:
                            print(f"Извлечено {len(subtasks)} подзадач из JSON")
                        if subtasks:
                            return subtasks
                except json.JSONDecodeError as e:
                    if debug_print:
                        print(f"Ошибка разбора JSON: {e}")

            # Способ 2: Ищем нумерованный список (как запасной вариант)
            pattern = re.compile(r'(?m)^\s*\d+[\.\)]\s+(.+)$')
            tasks = pattern.findall(answer)

            if tasks:
                if debug_print:
                    print(f"Извлечено {len(tasks)} подзадач из нумерованного списка")
                return [t.strip() for t in tasks]

            if debug_print:
                print("Не удалось извлечь подзадачи, возвращаем пустой список")

            return []

        def handle_recursion_limit_exceeded(error_text: str) -> str:
            """
            Стратегия обработки превышения глубины рекурсии

            :param error_text: Текст ошибки
            :return: Восстановленный ответ
            """
            output = error_text + "\n\n" + "Рассуждения были принудительно остановлены. Сейчас будет сгенерирован отчёт о проделанной работе вместе с ответом на ваше исходное сообщение.\n\n"
            if tracer:
                tracer.log(
                    depth=0,
                    phase="Recovery Attempt",
                    prompt=output + final_solution_text_generator_prompt,
                    extra={"reason": "recursion_limit_exceeded"},
                )

            # Получаем текущий уровень и ID задачи
            current_level = self.messages_meta_data.task_counter.get_order()
            current_task_id = self.messages_meta_data.task_counter.convert_to_str()

            # Получаем плейсхолдеры и локализуем промпт
            placeholders = get_prompt_placeholders(current_level, current_task_id)
            localized_prompt = localize_prompt(final_solution_text_generator_prompt, placeholders)

            recovery_response = self.response_from_LLM(
                user_message=output + localized_prompt, model_name=larger_model_name
            )
            if tracer:
                tracer.log(
                    depth=0,
                    phase="Recovery Response",
                    prompt=output + localized_prompt,
                    response=recovery_response,
                    extra={"status": "final_recovery"},
                )
            return recovery_response

        def solve_task(current_depth: int, skip_solution_generation: bool = False) -> str:
            """
            Рекурсивный процесс решения (и проверки) задачи.
            В зависимости от кода (а, б, в, г) либо завершаем,
            либо пересобираем, либо декомпозируем.

            :param current_depth: Текущая глубина рекурсии
            :param skip_solution_generation: Пропустить этап генерации решения (если уже есть интегрированное решение)
            :return: Решение задачи
            """
            nonlocal current_llm_calling_count
            current_llm_calling_count = max(current_llm_calling_count, current_depth)

            if current_llm_calling_count > self.max_llm_calling_count:
                error_msg = f"Превышена максимальная глубина рекурсии ({self.max_llm_calling_count})"
                if tracer:
                    tracer.log_error(depth=current_depth, error_msg=error_msg)
                raise RecursionError(error_msg)

            # Получаем текущий уровень и ID задачи
            current_level = self.messages_meta_data.task_counter.get_order()
            current_task_id = self.messages_meta_data.task_counter.convert_to_str()
            placeholders = get_prompt_placeholders(current_level, current_task_id)

            # ==================== (1) Генерация решения (пропускаем если указано) ============================
            if not skip_solution_generation:
                self.messages_meta_data.update_all_messages_statuses()
                self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

                # Локализуем промпт для генерации решения
                localized_start_solution_gen_prompt = localize_prompt(start_solution_gen_prompt, placeholders)

                start_time = time.time()
                solution = self.response_from_LLM(user_message=localized_start_solution_gen_prompt, model_name=larger_model_name)
                solution_time = time.time() - start_time
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Solution",
                        prompt=localized_start_solution_gen_prompt,
                        response=solution,
                        extra={"elapsed": solution_time},
                    )
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Solution",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Заменяем промпт на сокращенную версию
                self.messages_meta_data.safe_replace_prompt(
                    "Solution",
                    start_solution_gen_shortened_prompt,
                    debug_tracer=tracer,
                    depth=current_depth
                )
            else:
                # Логируем пропуск генерации
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Skip Solution Generation",
                        prompt="Использование интегрированного решения без повторной генерации",
                        extra={"skipped": True},
                    )

            # ====================== (2) Проверка решения ==========================
            self.messages_meta_data.update_all_messages_statuses()
            self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

            # Локализуем промпт для верификации решения
            localized_solution_verification_prompt = localize_prompt(solution_verification_prompt, placeholders)

            start_time = time.time()
            verification = self.response_from_LLM(user_message=localized_solution_verification_prompt, model_name=larger_model_name)
            verification_time = time.time() - start_time
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Solution Verification",
                    prompt=localized_solution_verification_prompt,
                    response=verification,
                    extra={"elapsed": verification_time},
                )
            self.messages_meta_data.add_metadata_in_last_message(
                command_number=0,
                message_type="Solution Verification",
                status=""
            )
            if tracer:
                tracer.set_messages_meta_data(self.messages_meta_data)

            # Заменяем промпт на сокращенную версию
            self.messages_meta_data.safe_replace_prompt(
                "Solution Verification",
                solution_verification_shortened_prompt,
                debug_tracer=tracer,
                depth=current_depth
            )

            # ===== (3) Выясняем, что делать дальше (а/б/в/г) =====================
            self.messages_meta_data.update_all_messages_statuses()
            self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

            # Локализуем промпт для определения действия
            localized_action_manager_prompt = localize_prompt(action_manager_prompt, placeholders)

            start_time = time.time()
            action_txt = self.response_from_LLM(user_message=localized_action_manager_prompt, model_name=model_name)
            action_time = time.time() - start_time
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Strategy Selection",
                    prompt=localized_action_manager_prompt,
                    response=action_txt,
                    extra={"elapsed": action_time},
                )
            self.messages_meta_data.add_metadata_in_last_message(
                command_number=0,
                message_type="Strategy Selection",
                status=""
            )
            if tracer:
                tracer.set_messages_meta_data(self.messages_meta_data)

            # Заменяем промпт на сокращенную версию
            self.messages_meta_data.safe_replace_prompt(
                "Strategy Selection",
                action_manager_shortened_prompt,
                debug_tracer=tracer,
                depth=current_depth
            )

            code = parsing_action_function(action_txt, debug_reasoning_print)
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Action Decision",
                    prompt=f"Выбрано действие: {code}",
                    extra={"action_code": code},
                )

            if debug_reasoning_print:
                print(f"[solve_task] Action code = {code}")

            # ---------- (а) Решение удовлетворяет критериям качества ------------
            if code == 'а':
                # ИЗМЕНЕНИЕ: Не применяем final_solution_text_generator_prompt здесь
                # Просто отмечаем решение как принятое и возвращаем

                # Логируем принятое решение
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Solution Accepted",
                        prompt="Решение принято без дополнительного форматирования",
                        extra={"action": "Решение принято"},
                    )

                # Добавляем метку для отслеживания принятого решения
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Accepted Solution",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Финальное форматирование будет применено в основной функции
                # после возврата из всей рекурсии
                return "accepted_solution"

            # -------------- (б) Лёгкие ошибки; пересобираем ----------------------
            elif code == 'б':
                self.messages_meta_data.update_all_messages_statuses()
                self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

                # Локализуем промпт для повторного решения
                localized_re_solve_unsuccessful_decision = localize_prompt(re_solve_unsuccessful_decision, placeholders)

                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Solution Retry",
                        prompt=localized_re_solve_unsuccessful_decision,
                        extra={"action": "Повторная попытка решения с исправлением"},
                    )
                self.context.add_user_message(text=localized_re_solve_unsuccessful_decision)
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Solution Retry",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Заменяем промпт на сокращенную версию перед повторным запуском
                self.messages_meta_data.safe_replace_prompt(
                    "Solution Retry",
                    re_solve_unsuccessful_decision_shortened_prompt,
                    debug_tracer=tracer,
                    depth=current_depth
                )

                # Здесь не пропускаем генерацию, т.к. нам нужно новое решение
                return solve_task(current_llm_calling_count + 3, skip_solution_generation=False)

            # ---------- (в) Решение неполное; докручиваем (продолжение) ----------
            elif code == 'в':
                self.messages_meta_data.update_all_messages_statuses()
                self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

                # Локализуем промпт для продолжения решения
                localized_continue_solution_prompt = localize_prompt(continue_solution_prompt, placeholders)

                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Solution Continuation",
                        prompt=localized_continue_solution_prompt,
                        extra={"action": "Продолжение незавершенного решения"},
                    )
                self.context.add_user_message(text=localized_continue_solution_prompt)
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Solution Continuation",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Заменяем промпт на сокращенную версию перед повторным запуском
                self.messages_meta_data.safe_replace_prompt(
                    "Solution Continuation",
                    continue_solution_shortened_prompt,
                    debug_tracer=tracer,
                    depth=current_depth
                )

                # Здесь не пропускаем генерацию, т.к. нам нужно продолжить решение
                return solve_task(current_llm_calling_count + 3, skip_solution_generation=False)

            # ---------- (г) Серьёзные ошибки; нужна декомпозиция -----------------
            elif code == 'г':
                self.messages_meta_data.update_all_messages_statuses()
                self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

                # Локализуем промпт для декомпозиции
                localized_decompose_task_prompt = localize_prompt(decompose_task_prompt, placeholders)

                start_time = time.time()
                decompose_answer = self.response_from_LLM(user_message=localized_decompose_task_prompt, model_name=larger_model_name)
                decompose_time = time.time() - start_time
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Task Decomposition",
                        prompt=localized_decompose_task_prompt,
                        response=decompose_answer,
                        extra={"elapsed": decompose_time},
                    )
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Task Decomposition",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Заменяем промпт на сокращенную версию
                self.messages_meta_data.safe_replace_prompt(
                    "Task Decomposition",
                    decompose_task_shortened_prompt,
                    debug_tracer=tracer,
                    depth=current_depth
                )

                subtasks = parsing_decompose_task_function(decompose_answer, debug_reasoning_print)
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Subtasks Extracted",
                        prompt=f"Извлечено {len(subtasks)} подзадач",
                        extra={"subtasks_count": len(subtasks), "subtasks": subtasks},
                    )

                if debug_reasoning_print:
                    print(f"[solve_task] Subtasks found: {subtasks}")

                # Добавляем проверку на пустой список подзадач
                if not subtasks:
                    if debug_reasoning_print:
                        print("[solve_task] Не удалось извлечь подзадачи, создаем стандартные")
                    # Создаем базовые подзадачи для обеспечения продолжения процесса
                    subtasks = [
                        "Анализ задачи: изучить условия, выявить ключевые элементы и требования, определить тип задачи и применимые методы решения",
                        "Разработка стратегии: определить наиболее эффективный подход к решению, выделить основные шаги, предусмотреть возможные трудности",
                        "Реализация решения: последовательно применить выбранную стратегию, формализовать каждый шаг, получить конкретный результат"
                    ]
                    if tracer:
                        tracer.log(
                            depth=current_depth,
                            phase="Default Subtasks",
                            prompt="Использованы стандартные подзадачи",
                            extra={"subtasks": subtasks},
                        )

                self.messages_meta_data.task_counter.increase_order()
                if tracer:
                    tracer.log_task_counter_state(current_depth, {"action": "increase_order"})
                    tracer.set_messages_meta_data(self.messages_meta_data)

                for i, sub in enumerate(subtasks):
                    if i > 0:
                        self.messages_meta_data.task_counter.increase_digit()
                        if tracer:
                            tracer.log_task_counter_state(current_depth, {"action": "increase_digit"})
                            tracer.set_messages_meta_data(self.messages_meta_data)

                    if tracer:
                        tracer.log(
                            depth=current_depth + 1,
                            phase="Subtask Start",
                            prompt=f"Подзадача {i+1}/{len(subtasks)}: {sub}",
                            extra={"subtask_index": i, "total_subtasks": len(subtasks)},
                        )
                    try:
                        recursion(task_text=sub, task_images=[], current_depth=current_llm_calling_count + 4)
                        if tracer:
                            tracer.log(
                                depth=current_depth + 1,
                                phase="Subtask Complete",
                                prompt=f"Подзадача {i+1}/{len(subtasks)} завершена",
                                extra={"subtask_index": i, "status": "success"},
                            )
                    except Exception as e:
                        if tracer:
                            tracer.log_error(
                                depth=current_depth + 1,
                                error_msg=f"Ошибка в подзадаче {i+1}: {str(e)}",
                                context=traceback.format_exc(),
                            )
                        raise

                self.messages_meta_data.task_counter.reduce_order()
                if tracer:
                    tracer.log_task_counter_state(current_depth, {"action": "reduce_order"})
                    tracer.set_messages_meta_data(self.messages_meta_data)

                self.messages_meta_data.update_all_messages_statuses()
                self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

                updated_placeholders = get_prompt_placeholders(current_level, current_task_id)
                localized_finish_task_after_solving_subtasks_prompt = localize_prompt(finish_task_after_solving_subtasks_prompt, updated_placeholders)

                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Subtasks Complete",
                        prompt="Все подзадачи решены, интеграция результатов",
                        extra={"action": "Завершение после подзадач"},
                    )
                self.context.add_user_message(text=localized_finish_task_after_solving_subtasks_prompt)
                self.messages_meta_data.add_metadata_in_last_message(
                    command_number=0,
                    message_type="Task Integration",
                    status=""
                )
                if tracer:
                    tracer.set_messages_meta_data(self.messages_meta_data)

                # Заменяем промпт на сокращенную версию
                self.messages_meta_data.safe_replace_prompt(
                    "Task Integration",
                    finish_task_after_solving_subtasks_shortened_prompt,
                    debug_tracer=tracer,
                    depth=current_depth
                )

                # ОПТИМИЗАЦИЯ: Вызываем solve_task с пропуском генерации решения,
                # поскольку интегрированное решение уже добавлено в контекст
                return solve_task(
                    current_depth=current_llm_calling_count,
                    skip_solution_generation=False
                )

        # --------------------------------------------------------------------------
        # Функция recursion(task_text, task_images), которая инициирует решение
        # подзадачи (фактически вызовет solve_task()) на новом уровне.
        # --------------------------------------------------------------------------
        def recursion(task_text: str, task_images: List[str], current_depth: int) -> str:
            """
            Рекурсивно решаем подзадачу task_text.
            Можно увеличить глубину TaskCounter, если хотим отмечать,
            что «подзадача» находится на новом уровне.

            :param task_text: Текст подзадачи
            :param task_images: Список изображений для подзадачи
            :param current_depth: Текущая глубина рекурсии
            :return: Решение подзадачи
            """
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Recursion Enter",
                    prompt=f"Вход в рекурсию для подзадачи: {task_text}",
                    extra={"has_images": bool(task_images)},
                )

            # Получаем текущий уровень и ID задачи для локализации промптов
            current_level = self.messages_meta_data.task_counter.get_order()
            current_task_id = self.messages_meta_data.task_counter.convert_to_str()
            placeholders = get_prompt_placeholders(current_level, current_task_id)

            # (а) Добавляем постановку подзадачи:
            self.messages_meta_data.update_all_messages_statuses()
            self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

            # Локализуем промпт task_statement с заменой {user_message} на текст задачи
            localized_task_statement = localize_prompt(task_statement_prompt, placeholders)
            localized_task_statement = localized_task_statement.replace("{user_message}", task_text)

            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Task Statement",
                    prompt=localized_task_statement,
                    extra={"original_task": task_text},
                )
            self.context.add_user_message(text=localized_task_statement, images=task_images)
            self.messages_meta_data.add_metadata_in_last_message(
                command_number=0,
                message_type="Task Statement",
                status=""
            )
            if tracer:
                tracer.set_messages_meta_data(self.messages_meta_data)

            # (b) Сформулировать теорию:
            self.messages_meta_data.update_all_messages_statuses()
            self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

            # Локализуем промпт для теории
            localized_theory_gen_prompt = localize_prompt(theory_gen_prompt, placeholders)

            start_time = time.time()
            theory_response = self.response_from_LLM(user_message=localized_theory_gen_prompt, model_name=model_name)
            theory_time = time.time() - start_time
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Theory",
                    prompt=localized_theory_gen_prompt,
                    response=theory_response,
                    extra={"elapsed": theory_time},
                )
            self.messages_meta_data.add_metadata_in_last_message(
                command_number=0,
                message_type="Theory",
                status=""
            )
            if tracer:
                tracer.set_messages_meta_data(self.messages_meta_data)

            # Заменяем промпт на сокращенную версию
            self.messages_meta_data.safe_replace_prompt(
                "Theory",
                theory_gen_shortened_prompt,
                debug_tracer=tracer,
                depth=current_depth
            )

            # (c) Выдвижение критериев:
            self.messages_meta_data.update_all_messages_statuses()
            self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

            # Локализуем промпт для критериев качества
            localized_quality_assessment_criteria_prompt = localize_prompt(quality_assessment_criteria_prompt, placeholders)

            start_time = time.time()
            criteria_response = self.response_from_LLM(
                user_message=localized_quality_assessment_criteria_prompt,
                model_name=larger_model_name
            )
            criteria_time = time.time() - start_time
            if tracer:
                tracer.log(
                    depth=current_depth,
                    phase="Quality Criteria",
                    prompt=localized_quality_assessment_criteria_prompt,
                    response=criteria_response,
                    extra={"elapsed": criteria_time},
                )
            self.messages_meta_data.add_metadata_in_last_message(
                command_number=0,
                message_type="Quality Criteria",
                status=""
            )
            if tracer:
                tracer.set_messages_meta_data(self.messages_meta_data)

            # Заменяем промпт на сокращенную версию
            self.messages_meta_data.safe_replace_prompt(
                "Quality Criteria",
                quality_assessment_criteria_shortened_prompt,
                debug_tracer=tracer,
                depth=current_depth
            )

            # (d) Собственно «решаем» (вызываем solve_task)
            try:
                # Увеличиваем глубину при входе в подзадачу
                solution_text = solve_task(current_llm_calling_count + 2)
                if tracer:
                    tracer.log(
                        depth=current_depth,
                        phase="Recursion Exit",
                        prompt=f"Выход из рекурсии для подзадачи",
                        response=solution_text[:200] + ("..." if len(solution_text) > 200 else ""),
                        extra={"status": "success"},
                    )
            except RecursionError as e:
                if tracer:
                    tracer.log_error(
                        depth=current_depth,
                        error_msg=f"{str(e)}\nКонтекст: подзадача '{task_text[:50]}...'",
                        context=traceback.format_exc(),
                    )
                raise
            except Exception as e:
                if tracer:
                    tracer.log_error(
                        depth=current_depth,
                        error_msg=f"Неожиданная ошибка: {str(e)} в подзадаче '{task_text[:50]}...'",
                        context=traceback.format_exc(),
                    )
                raise

            return solution_text

        # Проверяем существование файлов промптов
        try:
            # Загружаем все промпты
            # if not os.path.exists(os.path.join("..", "prompts")):
            #     raise FileNotFoundError(f"Директория с промптами не найдена: {os.path.join('..', 'prompts')}")

            # Проверяем наличие ключевых промптов
            if not main_recursive_decomposition_prompt:
                logging.warning("Основной промпт не найден, алгоритм может работать некорректно")

        except Exception as e:
            error_traceback = traceback.format_exc()
            logging.error(f"Ошибка при загрузке промптов: {str(e)}\n{error_traceback}")
            if debug_reasoning_print:
                print(f"Ошибка при загрузке промптов: {str(e)}")

        self.max_llm_calling_count = max_llm_calling_count
        current_llm_calling_count = 0

        # 1) Сохраняем «старый» контекст, клонируем если нужно
        saved_context = self.context.clone()

        # Логика, когда preserve_user_messages_post_analysis == True,
        # обычно: добавить user_message в «глобальный» контекст
        if preserve_user_messages_post_analysis:
            self.context.add_user_message(user_message, images)

        # Переключаемся, чтобы внутри использовать другой режим контекста (2),
        # или оставляем тот, что вам нужен:
        self.context = saved_context.clone()
        self.context.change_mod(2)

        # Инициализируем MessagesWithMetaData заново,
        # если вы хотите «чистое» хранение для данного цикла:
        self.messages_meta_data = MessagesWithMetaData(self.context.messages)

        # Добавим «главные» инструкции
        self.messages_meta_data.update_all_messages_statuses()
        self.messages_meta_data.rewrite_messages_content_with_updated_statuses()
        if tracer:
            tracer.log(
                depth=0,
                phase="Instruction",
                prompt=main_recursive_decomposition_prompt,
                extra={"step": "Добавление основной инструкции"},
            )
        self.context.add_user_message(text=main_recursive_decomposition_prompt)
        self.messages_meta_data.add_metadata_in_last_message(
            command_number=0,
            message_type="Instruction",
            status=""
        )
        if tracer:
            tracer.set_messages_meta_data(self.messages_meta_data)

        # ----------------------------------------------------------------------------
        # Запускаем recursion(...) для нашей основной задачи user_message
        # ----------------------------------------------------------------------------
        # Можно тоже вызвать self.messages_meta_data.task_counter.increase_order()
        # если хотите считать «Исходная задача» => «Уровень 1»,
        # но это уже ваш выбор.
        try:
            if tracer:
                tracer.log(
                    depth=0,
                    phase="Algorithm Start",
                    prompt=f"Начало рекурсивного алгоритма для задачи\n---\n{user_message[:200]}...",
                    extra={"has_images": bool(images)},
                )
            final_result = recursion(task_text=user_message, task_images=images, current_depth=current_llm_calling_count)
            if tracer:
                tracer.log(
                    depth=0,
                    phase="Algorithm Complete",
                    prompt="Алгоритм успешно завершен",
                    response=final_result[:200] + ("..." if len(final_result) > 200 else ""),
                    extra={"status": "success"},
                )
        except RecursionError as e:
            error_message = str(e)
            if tracer:
                tracer.log_error(
                    depth=0,
                    error_msg=f"Критическая ошибка: {error_message}",
                    context=traceback.format_exc(),
                )
            print(f"Критическая ошибка: {error_message}")
            final_result = handle_recursion_limit_exceeded(error_message)
            if tracer:
                tracer.log(
                    depth=0,
                    phase="Error Recovery",
                    prompt=f"Восстановление после ошибки: {error_message}",
                    response=final_result[:200] + ("..." if len(final_result) > 200 else ""),
                    extra={"status": "recovered"},
                )

        current_llm_calling_count = 0

        if tracer:
            tracer.log(
                depth=0,
                phase="Final Formatting",
                prompt="Применение финального форматирования к решению",
                extra={"status": "formatting"},
            )

        self.messages_meta_data.update_all_messages_statuses()
        self.messages_meta_data.rewrite_messages_content_with_updated_statuses()

        final_placeholders = {
            "level_indicator": "",
            "task_context": "основной задачей",
            "subtask_indicator": "",
            "parent_reference": "",
            "hierarchy_reminder": "Это основная задача, которую поставил пользователь.",
            "context_reminder": "Сформируй ответ, который будет полностью понятен пользователю."
        }

        localized_final_solution_text_generator_prompt = localize_prompt(
            final_solution_text_generator_prompt,
            final_placeholders
        )

        final_formatted_result = self.response_from_LLM(
            user_message=localized_final_solution_text_generator_prompt,
            model_name=larger_model_name
        )

        if tracer:
            tracer.log(
                depth=0,
                phase="Final Result",
                prompt=localized_final_solution_text_generator_prompt,
                response=final_formatted_result,
                extra={"status": "completed"},
            )

        self.messages_meta_data.add_metadata_in_last_message(
            command_number=0,
            message_type="Final Solution",
            status=""
        )
        if tracer:
            tracer.set_messages_meta_data(self.messages_meta_data)
            tracer.log_messages_context(self.messages_meta_data)
            tracer.log_context_to_file()

        return final_formatted_result
