import os
import json
import reflex as rx
from src.LLM_manager import ChatLLMAgent


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

    required_keys = [
        "openai_api_key",
        "openai_organization",
        "openrouter_api_key",
        "model_name",
        "larger_model_name",
        "temperature",
        "max_response_tokens",
        "max_total_tokens",
        "analysis_depth",
        "max_llm_calling_count",
        "use_openai_or_openrouter"
    ]
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

    def get_chat_agent(self) -> ChatLLMAgent:
        """Creates an instance of ChatLLMAgent."""
        return ChatLLMAgent(
            model_name=self.config["model_name"],
            openai_api_key=self.config["openai_api_key"],
            openai_organization=self.config["openai_organization"],
            openrouter_api_key=self.config["openrouter_api_key"],
            use_openai_or_openrouter=self.config["use_openai_or_openrouter"],
            mode=2,
            task_prompt="",
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
            question: The question to process.
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

        # response = agent.response_from_LLM(
        #     user_message=question,
        #     images=[],
        # )

        response = agent.response_from_LLM_with_hierarchical_recursive_decomposition(
            user_message=question,
            images=[],
            # model_name=MODEL_NAME,
            # larger_model_name=LARGER_MODEL_NAME,
            max_llm_calling_count=self.config["max_llm_calling_count"],
            preserve_user_messages_post_analysis=True,
            debug_reasoning_print=True
        )

        if response is not None:
            self.chats[self.current_chat][-1].answer += response
        else:
            response = ""
            self.chats[self.current_chat][-1].answer += response
            self.chats = self.chats
        self.chats = self.chats

        self.processing = False
