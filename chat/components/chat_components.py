import reflex as rx
from chat.state import State
from reflex_chakra import vstack, box, button, text, flex


def message(qa) -> rx.Component:
    """Отображение пары вопрос-ответ."""
    return vstack(
        box(
            text(qa.question, color=rx.color("mauve", 12), style={"white-space": "pre-wrap"}),
            align_self="flex-end",
            background_color=rx.color("violet", 4),
            padding="0.5em 1em",
            border_radius="10px",
            margin_bottom="0.5em",
        ),
        rx.cond(
            qa.answer != "",
            box(
                text(qa.answer, color=rx.color("mauve", 12), style={"white-space": "pre-wrap"}),
                align_self="flex-start",
                background_color=rx.color("mauve", 3),
                padding="0.5em 1em",
                border_radius="10px",
                margin_bottom="0.5em",
            ),
        ),
        width="100%",
    )


def chat() -> rx.Component:
    """Основной компонент чата."""
    return vstack(
        button(
            "Pynecone",
            background_color=rx.color("violet", 9),
            color=rx.color("mauve", 1),
            _hover={"background_color": rx.color("violet", 10)},
            margin="1em auto",
        ),
        rx.foreach(
            State.chats[State.current_chat],
            message,
        ),
        rx.cond(
            State.processing,
            text("Processing... Please wait.", color="blue"),
            text("")
        ),
        padding="1em",
        width="100%",
        flex="1",
        overflow_y="auto",
    )


def action_bar() -> rx.Component:
    """Панель ввода с полем и кнопкой отправки."""
    return rx.form(
        flex(
            rx.text_area(
                id="question",
                placeholder="Введите сообщение...",
                value=State.question,
                on_change=State.set_question,
                _on_key_down="""
                    // Отладка: логируем нажатие клавиш
                    console.log('Key pressed:', event.key, 'Shift:', event.shiftKey, 'Processing:', $state.processing);
                    console.log('Current question value:', $state.question);

                    // Проверяем, нажата ли клавиша Enter
                    if (event.key === 'Enter') {
                        // Shift+Enter: добавляет перенос строки
                        if (event.shiftKey) {
                            event.preventDefault();
                            $state.add_new_line();
                            console.log('Shift+Enter detected - adding new line');
                        }
                        // Enter (без Shift): отправляет запрос через State.submit_question
                        else {
                            event.preventDefault();
                            console.log('Enter detected - submitting question');
                            $state.submit_question();
                        }
                    }
                """,
                width="100%",
                min_height="40px",
                max_height="100px",
                padding="0.5em",
                resize="vertical",
                background_color=rx.color("mauve", 3),
                color=rx.color("mauve", 12),
                border="none",
                _focus={
                    "border": f"1px solid {rx.color('violet', 6)}",
                    "outline": "none",
                },
            ),
            button(
                "Отправить",
                type_="submit",
                background_color=rx.color("violet", 9),
                color=rx.color("mauve", 1),
                _hover={"background_color": rx.color("violet", 10)},
                padding="0.5em 1em",
                is_loading=State.processing,
                is_disabled=State.processing,
            ),
            width="100%",
            spacing="1em",
            padding="1em",
            align_items="stretch",
        ),
        id="question-form",
        on_submit=[
            State.process_question,
            rx.set_value("question", ""),  # Clear the input field immediately after submission
        ],
        width="100%",
        reset_on_submit=True,
    )