# Chat LLM with Hierarchical Recursive Decomposition (HRD)

Фреймворк для решения сложных задач с использованием больших языковых моделей (LLM) через автоматическую декомпозицию на подзадачи, рекурсивное решение и проверку качества на каждом этапе.

## Основная концепция HRD

Hierarchical Recursive Decomposition (HRD) - это алгоритм, расширяющий возможности языковых моделей при решении сложных задач путём:

1. Автоматической декомпозиции задачи на подзадачи
2. Формализации критериев качества решения
3. Строгой верификации решений на соответствие критериям
4. Адаптивного выбора стратегии на основе результатов проверки

## Архитектура алгоритма

Алгоритм HRD реализует следующую последовательность:

1. **Формулировка задачи** - анализ исходной задачи
2. **Теоретическая база** - определение применимых концепций и методов
3. **Критерии качества** - формулировка критериев проверки решения 
4. **Генерация решения** - попытка решить задачу
5. **Верификация решения** - проверка на соответствие критериям
6. **Выбор стратегии** на основе результатов верификации:
   - **a**: Решение удовлетворяет критериям → формирование финального ответа
   - **б**: Решение содержит легко устранимые ошибки → повторное решение 
   - **в**: Решение неполное, но корректное → продолжение решения
   - **г**: Решение содержит серьезные ошибки → декомпозиция на подзадачи
7. **При выборе варианта "г"**:
   - Разбиение на подзадачи
   - Рекурсивное решение каждой подзадачи
   - Интеграция результатов

### Система промптов

Алгоритм HRD использует набор специализированных промптов для различных этапов:

- **main_recursive_decomposition_prompt.txt** - основная инструкция для HRD
- **task_statement_prompt.txt** - формулировка задачи
- **theory_gen_prompt.txt** - построение теоретической базы
- **quality_assessment_criteria_prompt.txt** - формирование критериев качества
- **start_solution_gen_prompt.txt** - генерация решения
- **solution_verification_prompt.txt** - верификация решения
- **action_manager_prompt.txt** - определение дальнейших действий
- **decompose_task_prompt.txt** - разбиение на подзадачи
- **finish_task_after_solving_subtasks_prompt.txt** - интеграция результатов

В папке **shortened_prompts/** хранятся компактные версии этих промптов, которые используются для оптимизации контекста после выполнения соответствующего этапа.

## Ключевые компоненты

### ChatLLMAgent

Основной класс, реализующий взаимодействие с LLM и алгоритм HRD. Ключевые методы:
- `response_from_LLM()` - стандартный запрос к модели
- `response_from_LLM_with_hierarchical_recursive_decomposition()` - применение HRD

### MessageContext

Класс для управления контекстом сообщений с тремя режимами работы (задается через параметр `mode`):
- `1`: Очистка контекста перед каждым новым сообщением пользователя
- `2`: Сохранение контекста между запросами, системный промпт только в начале 
- `3`: Сохранение контекста, добавление системного промпта перед каждым сообщением пользователя

### MessagesWithMetaData

Класс для управления метаданными и иерархией сообщений в контексте:
- Отслеживает иерархическую структуру задач и подзадач
- Помечает сообщения статусами (in_progress, resolved и т.д.)
- Оптимизирует контекст, заменяя полные промпты сокращенными версиями
- Поддерживает клонирование контекста для рекурсивного решения

### DebugTracer

Инструмент для детальной трассировки выполнения алгоритма HRD:
- Логирует все этапы решения с временными метками
- Сохраняет промпты и ответы модели
- Отслеживает иерархию задач и подзадач
- Создает JSONL файлы с полной информацией о выполнении
- Предоставляет форматированный вывод в консоль


## Настройка и запуск

### Конфигурация

Создайте `config.json` в корневой директории:

```json
{
    "openai_api_key": "sk-your-openai-key",
    "openai_organization": "org-your-org-id", 
    "openrouter_api_key": "sk-or-your-openrouter-key",
    "model_name": "gpt-4o",
    "larger_model_name": "gpt-4o", 
    "temperature": 0.0,
    "max_response_tokens": 4095,
    "max_total_tokens": 120000,
    "analysis_depth": 5,
    "max_llm_calling_count": 100,
    "use_openai_or_openrouter": "openai"
}
```

### Пример использования

```python
import json
from src.LLM_manager import ChatLLMAgent

with open("../config.json", "r") as file:
    config = json.load(file)

OPENAI_API_KEY = config["openai_api_key"]
OPENAI_ORGANIZATION = config["openai_organization"]
OPENROUTER_API_KEY = config["openrouter_api_key"]
TEMPERATURE = config.get("temperature", 0.0)
MAX_RESPONSE_TOKENS = config.get("max_response_tokens", 4096)
MAX_TOTAL_TOKENS = config.get("max_total_tokens", 60000)
MODEL_NAME = config.get("model_name", "gpt-4o")
USE_OPENAI_OR_OPENROUTER = config.get("use_openai_or_openrouter", "openai")
ANALYSIS_DEPTH = config.get("analysis_depth", 5)
MAX_LLM_CALLING_COUNT = config.get("max_llm_calling_count", 5)

# Проверяем наличие larger_model_name, если его нет - используем MODEL_NAME
LARGER_MODEL_NAME = config.get("larger_model_name", MODEL_NAME)

# Замечание: исследования показывают, что не следует даже в MODEL_NAME использовать модель
# с менее чем 100 миллиардами параметров (LARGER_MODEL_NAME >= MODEL_NAME > 100B params)

agent_IoT = ChatLLMAgent(
    model_name=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
    openrouter_api_key=OPENROUTER_API_KEY,
    use_openai_or_openrouter=USE_OPENAI_OR_OPENROUTER,
    mode=2,
    task_prompt="Ты педантичный математик с безупречной точностью. "
                "Всегда проверяй каждый шаг вычислений дважды, объясняй каждое преобразование подробно "
                "и никогда не пропускай промежуточные выкладки. Арифметические ошибки недопустимы.",
    max_total_tokens=MAX_TOTAL_TOKENS,
    max_response_tokens=MAX_RESPONSE_TOKENS,
    temperature=TEMPERATURE
)

# Разделяем контексты через отдельных агентов
agent_HRD = agent_IoT.clone()

# user_prompt = "Сколько будет 2+2?"
user_prompt = "Вычислить $\lim_{n \to \infty} \sin \frac{\pi}{n} \sum_{k=1}^{n} \frac{1}{2 + \cos \frac{\pi k}{n}}$."
# user_prompt = "В круге радиусом 5 хорды \(AB\) и \(CD\) пересекаются в точке \(E\). Если точка \(B\) лежит на малой дуге \(AD\), \(BC = 6\), и \(AD\) — единственная хорда, начинающаяся в \(A\), которая пересекается пополам линией \(BC\), найдите синус центрального угла малой дуги \(AB\), выраженный в виде \(m/n\) в несократимом виде, и вычислите \(mn\).    "
# user_prompt = "$9$ членов бейсбольной команды пошли в кафе-мороженое после своей игры. Каждый игрок взял по одному рожку мороженого с шоколадным, ванильным или клубничным вкусом. По крайней мере один игрок выбрал каждый из вкусов, причем количество игроков, выбравших шоколадное мороженое, было больше, чем количество выбравших ванильное, которое в свою очередь было больше, чем количество выбравших клубничное. Пусть $N$ — число различных вариантов распределения вкусов между игроками, удовлетворяющих этим условиям. Найдите остаток от деления $N$ на $1000$."

response = agent_IoT.response_from_LLM(
    user_message=user_prompt,
    images=[],
)

print("Ответ бота БЕЗ иерархической рекурсивной декомпозиции:")
print(response)


response = agent_HRD.response_from_LLM_with_hierarchical_recursive_decomposition(
    user_message=user_prompt,
    images=[],
    model_name=MODEL_NAME,  # Используется для более простых целей (можно указать чуть более слабую модель)
    larger_model_name=LARGER_MODEL_NAME,  # Используется для сложных целей
    max_llm_calling_count=MAX_LLM_CALLING_COUNT,
    preserve_user_messages_post_analysis=True,
    debug_reasoning_print=True
)

print("Ответ бота С иерархической рекурсивной декомпозицией:")
print(response)
```

## Параметры и опции

### Параметры ChatLLMAgent

- **model_name**: Название модели для API
- **openai_api_key/openai_organization**: Для OpenAI API
- **openrouter_api_key**: Для OpenRouter API
- **use_openai_or_openrouter**: Выбор провайдера API ("openai" или "openrouter")
- **mode**: Режим работы контекста (1, 2 или 3)
- **task_prompt**: Системный промпт для определения роли модели
- **max_total_tokens**: Максимальное количество токенов в контексте
- **max_response_tokens**: Максимальное количество токенов в ответе
- **temperature**: Параметр креативности (0.0 - 1.0)

### Параметры метода HRD

- **user_message**: Задача для решения
- **images**: Список изображений (если применимо)
- **model_name**: Модель для базовых операций
- **larger_model_name**: Модель для критических операций
- **max_llm_calling_count**: Ограничение глубины рекурсии
- **preserve_user_messages_post_analysis**: Сохранять ли запрос в глобальном контексте
- **debug_reasoning_print**: Включение отладочного вывода

## Запуск как веб-приложение (Reflex)

Данный проект является форком демонстрационного приложения на Reflex. Чтобы запустить весь интерфейс как веб-приложение, выполните следующие действия:

1. Убедитесь, что у вас установлены:  
   • Python 3.11+  
   • Node.js 12.22.0+  
   • Необходимые зависимости из requirements.txt  

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Заполните файл config.json, как описано выше

4. Инициализируйте и запустите приложение Reflex:
   ```bash
   reflex init
   reflex run
   ```
   Приложение будет доступно по адресу http://localhost:3000.

<div align="center">
<img src="./docs/demo.gif" alt="icon"/>
</div>
