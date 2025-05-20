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
