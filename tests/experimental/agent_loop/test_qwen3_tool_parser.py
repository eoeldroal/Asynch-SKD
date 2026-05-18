import json

import pytest

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema


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


@pytest.mark.asyncio
async def test_qwen3_tool_parser_parses_array_parameter_as_json_before_eval():
    parser = Qwen3XMLToolParser(
        _Tokenizer(
            "<tool_call>\n"
            "<function=computer>\n"
            "<parameter=actions>\n"
            '[{"action_type":"TYPING","text":"mkdir -p tmp","clear":false,"enter":false}]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
    )
    tool_schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "computer",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action_type": {"type": "string"},
                                    "text": {"type": "string"},
                                    "clear": {"type": "boolean"},
                                    "enter": {"type": "boolean"},
                                },
                                "required": ["action_type"],
                            },
                        }
                    },
                    "required": ["actions"],
                },
            },
        }
    )

    _, tool_calls = await parser.extract_tool_calls([1], tools=[tool_schema])

    assert parser.last_parse_error is None
    assert len(tool_calls) == 1
    parsed_args = json.loads(tool_calls[0].arguments)
    assert parsed_args["actions"][0]["action_type"] == "TYPING"
    assert parsed_args["actions"][0]["clear"] is False
    assert parsed_args["actions"][0]["enter"] is False
