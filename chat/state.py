import os
import json
import reflex as rx
from src.LLM_manager import ChatGPTAgent


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Intros": [],
}


def load_config():
    """Load configuration from a JSON file and check for required keys."""
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON in {config_path}: {e}")

    required_keys = ["api_key", "organization", "model_name", "temperature", "max_response_tokens", "max_total_tokens", "analysis_depth"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config parameters: {', '.join(missing_keys)}")

    return config


class State(rx.State):
    """The app state."""
    # Loading the configuration when initializing the class
    config = load_config()

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    def get_chat_agent(self) -> ChatGPTAgent:
        """Creates an instance of ChatGPTAgent."""
        return ChatGPTAgent(
            api_key=self.config["api_key"],
            organization=self.config["organization"],
            model_name=self.config["model_name"],
            mode=2,
            task_prompt="""You are "ChatGPT with chain of reasoning" (If they ask, then give them your full name and chain of reasoning can be translated into the user's language). This enhanced version of ChatGPT applies a step-by-step chain-of-reasoning method before each response. While this increases response time, it significantly boosts the quality of answers, allowing the chatbot to tackle much more complex tasks.
Instruction: When answering my questions, always use a step-by-step reasoning approach to fully understand, analyze, and provide an accurate answer.
Chain of reasoning reasoning or step-by-step is a method where you break down each question, asking yourself smaller, clarifying questions and answering them sequentially. This ensures a correct, well-verified final answer with zero errors. First, outline the questions that need to be addressed, then proceed to answer them. Use this method consistently to avoid providing inaccurate information, write error-free code, and identify any bugs in the code, if present.
If you're asked to translate text, focus on conveying the intended meaning rather than providing a literal translation. The translation should be professionally oriented, accessible to a technically savvy reader, and preserve complex constructions when they hold important meaning. For technical texts, if professional terms appear in English, keep them in English; if a Russian equivalent exists, include it in parentheses before the English term. You are highly experienced in IT, enabling you to translate advanced technical materials, particularly those related to databases and distributed computing.
If the questions are in advanced mathematics, imagine you are a professor in mathematical analysis. Guide through topics step-by-step with strict mathematical precision, providing examples and detailed explanations. Anticipate possible mistakes to help prevent misinterpretations, and clarify each statement to ensure it is both accessible and mathematically rigorous.
The final answer is always given in the user's language.
If they do not ask otherwise, then give answers using MarkDown markup.
""",
            max_total_tokens=self.config["max_total_tokens"],
            max_response_tokens=self.config["max_response_tokens"],
            temperature=self.config["temperature"]
        )

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.openai_process_question

        async for value in model(question):
            yield value

    async def openai_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """
        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        agent = self.get_chat_agent()

        for qa in self.chats[self.current_chat]:
            agent.context.add_user_message(qa.question)
            if qa.answer != "":
                agent.context.add_assistant_message(qa.answer)

        # response = agent.response_from_chat_GPT_with_chain_of_reasoning(analysis_depth=self.config["analysis_depth"], user_message=question, images=[], preserve_user_messages_post_analysis=True)
        # Replaced it so that the 'analysis_depth' could be changed right during operation
        response = agent.response_from_chat_GPT_with_chain_of_reasoning(
            analysis_depth=load_config()["analysis_depth"], user_message=question, images=[], preserve_user_messages_post_analysis=True, debug_reasoning_print=True)

        if response is not None:
            self.chats[self.current_chat][-1].answer += response
        else:
            response = ""
            self.chats[self.current_chat][-1].answer += response
            self.chats = self.chats
        self.chats = self.chats

        self.processing = False
