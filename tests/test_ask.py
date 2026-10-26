from types import SimpleNamespace
from unittest.mock import Mock

from app import ask


def test_ask_displays_input_and_output_token_counts(capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
        text="answer",
    )
    chat = Mock()
    chat.send_message.return_value = response

    assert ask.ask(chat, "question") == "answer"
    chat.send_message.assert_called_once_with("question")

    assert "[トークン数] 入力: 12, 出力: 7" in capsys.readouterr().out


def test_ask_displays_response_time(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="answer")
    chat = Mock()
    chat.send_message.return_value = response
    timestamps = iter([10.0, 11.234])
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(timestamps))

    assert ask.ask(chat, "question") == "answer"

    assert "[応答時間] 1.23 秒" in capsys.readouterr().out


def test_ask_reports_missing_usage_metadata(capsys):
    response = SimpleNamespace(usage_metadata=None, text="answer")
    chat = Mock()
    chat.send_message.return_value = response

    assert ask.ask(chat, "question") == "answer"

    assert "[トークン数] 入力: 不明, 出力: 不明" in capsys.readouterr().out


def test_main_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    chat = Mock()
    chat.send_message.side_effect = [
        SimpleNamespace(usage_metadata=None, text="first answer"),
        SimpleNamespace(usage_metadata=None, text="follow-up answer"),
    ]
    client = Mock()
    client.chats.create.return_value = chat
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    monkeypatch.setattr("builtins.input", Mock(side_effect=["follow-up", "exit"]))

    ask.main()

    client.chats.create.assert_called_once_with(model=ask.MODEL)
    assert [call.args[0] for call in chat.send_message.call_args_list] == [
        "first question",
        "follow-up",
    ]
    output = capsys.readouterr().out
    assert "first answer" in output
    assert "follow-up answer" in output
