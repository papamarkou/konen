from konen import LLMResponse


def test_llm_response_preserves_fields() -> None:
    response = LLMResponse(
        text="Posterior probability is 0.82.",
        metadata={"engine_uid": "gpt-analysis"},
        provider_state={"response_id": "resp_01HXYZ123"},
    )

    assert response.text == "Posterior probability is 0.82."
    assert response.metadata == {"engine_uid": "gpt-analysis"}
    assert response.provider_state == {"response_id": "resp_01HXYZ123"}


def test_llm_response_independent_defaults() -> None:
    first = LLMResponse(text="First")
    second = LLMResponse(text="Second")

    first.metadata["key"] = "value"
    first.provider_state["state"] = "value"

    assert second.metadata == {}
    assert second.provider_state == {}


def test_llm_response_deep_copies_inputs() -> None:
    metadata = {"token_usage": {"input_token_count": 10}}
    provider_state = {"messages": [{"role": "assistant", "content": "Hello"}]}

    response = LLMResponse(text=None, metadata=metadata, provider_state=provider_state)

    metadata["token_usage"]["input_token_count"] = 99
    provider_state["messages"][0]["content"] = "Changed"

    assert response.text is None
    assert response.metadata == {"token_usage": {"input_token_count": 10}}
    assert response.provider_state == {
        "messages": [{"role": "assistant", "content": "Hello"}]
    }
