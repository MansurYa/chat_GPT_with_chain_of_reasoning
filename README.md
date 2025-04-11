# Chat_GPT_with_chain_of_reasoning

Данный проект представляет собой расширенную версию "ChatGPT" с методом пошагового рассуждения ("chain of reasoning"), что позволяет боту глубже анализировать вопросы и давать максимально точные ответы. Основной функционал проекта реализован в двух файлах:

• config.json – файл конфигурации, где указываются ключи и параметры для работы с OpenAI API.  
• chat_GPT_manager.py – основной программный код, включающий классы для управления контекстом и общения с моделью GPT.

Всё остальное в репозитории – это графический интерфейс (UI), заимствованный из форкнутого проекта, который позволяет запускать данное решение как веб-приложение на основе Reflex.

--------------------------------------------------------------------------------

## Основной функционал

1. **MessageContext**  
   Удобный класс для управления контекстом GPT-модели. Содержит 3 режима (mod) для контроля того, как сообщения пользователя и системные сообщения хранятся и переиспользуются между запросами.

2. **ChatGPTAgent**  
   Класс высокого уровня для взаимодействия с API-моделей (OpenAI GPT). Управляет отправкой сообщений, обработкой ответов и учётом лимитов токенов.

3. **response_from_chat_GPT_with_chain_of_reasoning**  
   Главный метод для выполнения пошагового (глубокого) рассуждения. Модель сначала строит «дорожную карту» (последовательность внутренних вопросов и подзадач), а затем пошагово формирует ответ, добиваясь более точных результатов.

--------------------------------------------------------------------------------

## Быстрый пример использования в Python

Ниже пример кода, где создаётся агент, а затем вызывается метод с цепочкой рассуждений:

```python
import json
from src.chat_GPT_manager import ChatGPTAgent

with open("../config.json", "r") as file:
   config = json.load(file)

API_KEY = config["api_key"]
ORGANIZATION = config["organization"]
TEMPRATURE = config.get("temperature", 0.0)
MAX_RESPONSE_TOKENS = config.get("max_response_tokens", 4096)  
MAX_TOTAL_TOKENS = config.get("max_total_tokens", 30000)
MODEL_NAME = config.get("model_name", "gpt-4o")
ANALYSIS_DEPTH = config.get("analysis_depth", 5)

# Инициализируем агента:
agent = ChatGPTAgent(
    api_key=API_KEY,
    organization=ORGANIZATION,
    model_name=MODEL_NAME,
    mode=2,
    task_prompt="You are an advanced ChatGPT with chain of reasoning.",
    max_total_tokens=MAX_TOTAL_TOKENS,           
    max_response_tokens=MAX_RESPONSE_TOKENS,     
    temperature=TEMPRATURE                       
)

# Выполняем запрос с пошаговым рассуждением
user_prompt = "Помоги мне составить подробный план изучения Data Science."
response = agent.response_from_chat_GPT_with_chain_of_reasoning(
    analysis_depth=ANALYSIS_DEPTH,
    user_message=user_prompt,
    images=[],
    preserve_user_messages_post_analysis=True
)

print("Ответ бота с цепочкой рассуждений:")
print(response)
```

В данном примере:  
• «analysis_depth» управляет количеством итераций для снижения вероятности пропустить детали задачи.  
• «preserve_user_messages_post_analysis» = True хранит исходный вопрос пользователя в итоговом контексте.  

--------------------------------------------------------------------------------

## Запуск как веб-приложение (Reflex)

Данный проект является форком демонстрационного приложения на Reflex. Чтобы запустить весь интерфейс как веб-приложение, выполните следующие действия:

1. Убедитесь, что у вас установлены:  
   • Python 3.7+  
   • Node.js 12.22.0+  
   • Необходимые зависящие пакеты, перечисленные в requirements.txt  

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Заполните файл config.json (или укажите ключи в переменных окружения).  
   Образец config.json:  
   ```json
   {
       "api_key": "sk-...YOURKEY...",
       "organization": "org-...",
       "model_name": "gpt-4",
       "temperature": 0.0,
       "max_response_tokens": 4095,
       "max_total_tokens": 30000,
       "analysis_depth": 5
   }
   ```

4. Инициализируйте и запустите приложение Reflex:
   ```bash
   reflex init
   reflex run
   ```
   Приложение будет доступно по адресу http://localhost:3000.

--------------------------------------------------------------------------------

## Запуск, не используя Reflex

Если вам не нужен веб-интерфейс, можно просто импортировать и использовать класс ChatGPTAgent в своей Python-среде, как показано в разделе «Быстрый пример использования». Скрипт chat_GPT_manager.py содержит все нужные методы.

--------------------------------------------------------------------------------

## Важные файлы

• config.json – параметры для настройки API GPT, глубины анализа и прочего.  
• src/chat_GPT_manager.py – содержит:  
  – Класс MessageContext (управление контекстом и режимами).  
  – Класс ChatGPTAgent (интеграция с моделью, учёт токенов, запросы).  
  – Метод response_from_chat_GPT_with_chain_of_reasoning (ключевой метод для глубокого пошагового анализа).  

--------------------------------------------------------------------------------

## Reflex Chat App

Часть функционала (UI, процессы запуска) унаследована из форкнутого репозитория, где использовался Reflex для создания и стилизации веб-интерфейса. Инструкции по установке Reflex, переменным окружения и запуску взяты оттуда. Оригинальное DEMO, дизайн и базовая идея веб-приложения принадлежат Reflex.

<div align="center">
<img src="./docs/demo.gif" alt="icon"/>
</div>
