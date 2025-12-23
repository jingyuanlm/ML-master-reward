"""Backend for LiteLLM API."""

import json
import logging
import time

from backend.backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import litellm
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

LITELLM_TIMEOUT_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.InternalServerError,
)



def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    cfg: Config = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    message_print = messages[0]["content"]
    print(f"\033[31m{message_print}\033[0m")
    
    completion = backoff_create(
        litellm.completion,
        LITELLM_TIMEOUT_EXCEPTIONS,
        messages=messages,
        api_base=cfg.agent.feedback.base_url,
        api_key=cfg.agent.feedback.api_key,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
        print(f"\033[32m{output}\033[0m")
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
            print(f"\033[32m{output}\033[0m")
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
