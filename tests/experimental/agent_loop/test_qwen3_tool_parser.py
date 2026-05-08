import pytest

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser


class _Tokenizer:
    def __init__(self, text: str):
        self._text = text

    def decode(self, response_ids):
        del response_ids
        return self._text


@pytest.mark.asyncio
async def test_qwen3_tool_parser_records_actions_json_parse_error():
    parser = Qwen3XMLToolParser(
        _Tokenizer(
            "<tool_call>\n"
            "<function=computer>\n"
            "<parameter=actions>\n"
            '[{"action_type":"CLICK","x":123,"y":148]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
    )

    _, tool_calls = await parser.extract_tool_calls([1], tools=None)

    assert tool_calls == []
    assert parser.last_parse_error is not None
    assert parser.last_parse_error.kind == "actions_json_malformed"
    assert parser.last_parse_error.message == "the actions JSON is malformed."


@pytest.mark.asyncio
async def test_qwen3_tool_parser_records_incomplete_tool_tag_error():
    parser = Qwen3XMLToolParser(
        _Tokenizer(
            "<tool_call>\n"
            "<function=computer\n"
            "<parameter=actions>\n"
            '[{"action_type":"CLICK","x":123,"y":148}]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
    )

    _, tool_calls = await parser.extract_tool_calls([1], tools=None)

    assert tool_calls == []
    assert parser.last_parse_error is not None
    assert parser.last_parse_error.kind == "tool_tag_incomplete"
    assert parser.last_parse_error.message == "a tool-call tag is incomplete."
