"""Adapter Anthropic com cliente mockado: escolha entre modo strict e lenient, sem rede."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx2 as httpx
import pytest
from pydantic import BaseModel, create_model

from doc_read_ai.domain.analysis import ValidationVerdict
from doc_read_ai.domain.config import AgentConfig, ProviderConfig
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.infrastructure.llm.anthropic_adapter import (
    MAX_UNION_PARAMS,
    AnthropicLLM,
    schema_complexity,
)

VERDICT = ValidationVerdict(is_valid_document=True, reason="ok", legibility="high")


def big_model(**extra_fields: tuple) -> type[BaseModel]:
    """Modelo com mais campos anuláveis do que structured outputs aceita (como uma NF-e)."""
    fields = {f"campo_{i}": (str | None, None) for i in range(MAX_UNION_PARAMS + 4)}
    return create_model("NotaGrande", **fields, **extra_fields)


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def parsed_response(parsed: BaseModel) -> SimpleNamespace:
    response = text_response(parsed.model_dump_json())
    response.parsed_output = parsed
    return response


def make_llm(**messages_methods: AsyncMock) -> tuple[AnthropicLLM, MagicMock]:
    client = MagicMock()
    client.beta.messages.parse = messages_methods.get("parse", AsyncMock())
    client.beta.messages.create = messages_methods.get("create", AsyncMock())
    agent = AgentConfig(name="extractor", provider="anthropic", model="claude-haiku-5-5")
    return AnthropicLLM(agent, ProviderConfig(), client=client), client


async def generate(llm: AnthropicLLM, schema: type[BaseModel]) -> BaseModel:
    return await llm.generate_structured(
        system_prompt="sys", instruction="extraia", output_schema=schema
    )


def test_schema_complexity_counts_unions_and_optionals():
    unions, optional = schema_complexity(big_model().model_json_schema())
    assert unions == MAX_UNION_PARAMS + 4
    assert optional == MAX_UNION_PARAMS + 4


async def test_small_schema_uses_strict_structured_outputs():
    llm, client = make_llm(parse=AsyncMock(return_value=parsed_response(VERDICT)))

    assert await generate(llm, ValidationVerdict) == VERDICT
    assert client.beta.messages.parse.await_args.kwargs["output_format"] is ValidationVerdict
    client.beta.messages.create.assert_not_called()


async def test_big_schema_uses_lenient_mode_with_schema_in_prompt():
    model = big_model()
    create = AsyncMock(return_value=text_response('```json\n{"campo_0": "abc"}\n```'))
    llm, client = make_llm(create=create)

    result = await generate(llm, model)

    assert result.campo_0 == "abc" and result.campo_1 is None
    client.beta.messages.parse.assert_not_called()
    prompt = create.await_args.kwargs["messages"][0]["content"][-1]["text"]
    assert "JSON Schema" in prompt and "campo_0" in prompt
    assert "output_format" not in create.await_args.kwargs


async def test_lenient_mode_asks_model_to_fix_invalid_json():
    model = big_model(total=(float | None, None))
    create = AsyncMock(
        side_effect=[text_response('{"total": "não é número"}'), text_response('{"total": 10.5}')]
    )
    llm, _ = make_llm(create=create)

    result = await generate(llm, model)

    assert result.total == 10.5
    retry_messages = create.await_args_list[1].kwargs["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert "validação" in retry_messages[-1]["content"]


async def test_lenient_mode_gives_up_after_one_fix():
    create = AsyncMock(return_value=text_response("isso não é json"))
    llm, _ = make_llm(create=create)

    with pytest.raises(LLMError, match="mesmo após correção"):
        await generate(llm, big_model())
    assert create.await_count == 2


async def test_strict_falls_back_to_lenient_when_server_rejects_schema_complexity():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    too_complex = anthropic.BadRequestError(
        "Schema is too complex for compilation.",
        response=httpx.Response(400, request=request),
        body=None,
    )
    create = AsyncMock(return_value=text_response(VERDICT.model_dump_json()))
    llm, _ = make_llm(parse=AsyncMock(side_effect=too_complex), create=create)

    assert await generate(llm, ValidationVerdict) == VERDICT
    create.assert_awaited_once()


async def test_other_bad_requests_are_not_retried():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    bad = anthropic.BadRequestError(
        "model not found", response=httpx.Response(400, request=request), body=None
    )
    create = AsyncMock()
    llm, _ = make_llm(parse=AsyncMock(side_effect=bad), create=create)

    with pytest.raises(LLMError, match="model not found"):
        await generate(llm, ValidationVerdict)
    create.assert_not_called()
