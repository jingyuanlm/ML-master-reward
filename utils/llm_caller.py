import os
from typing import List, Dict, Union, Optional, Any
import litellm
import time
import logging
logger = logging.getLogger("ml-master")


class LLM:
    """
    Encapsulate the VLLM-based LLM class to invoke the self-hosted VLLM model via LiteLLM.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        model_name: str = "default-model",
        temperature: float = 0.7,
        max_tokens: int = 16384,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        retry_time: int = 20,
        delay_time: int = 3,
    ):
        """
        Initialize the VLLM LLM class.
        
        Args:
            base_url: The URL of the VLLM service.
            api_key: API key (generally not important when self-hosted).
            model_name: Name of the model (generally not important when self-hosted).
            temperature: Temperature parameter to control output randomness.
            max_tokens: Maximum number of tokens to generate.
            stop_tokens: List of tokens that signal the end of generation.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens
        self.retry_time = retry_time
        self.delay_time = delay_time
    
    def generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the LiteLLM API
            
        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns the streaming response object
        """
        
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            "api_base": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = litellm.completion(**params)
                
                if stream:
                    return response
                
                return response.choices[0].message
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)

    def stream_generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        delay_time:int = 1,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            **kwargs: Additional parameters passed to the LiteLLM API
            
        Returns:
            Returns the streaming response object
        """
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            "api_base": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = litellm.completion(**params)
                full_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        full_text += chunk.choices[0].delta.content
                    return response
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)
                
    
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the LiteLLM API

        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns a streaming response object
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # Convert prompt to messages format for LiteLLM
        messages = [{"role": "user", "content": prompt}]
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            "api_base": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        response = litellm.completion(**params)
        
        if stream:
            return response
        
        return response.choices[0].message.content
    
    def stream_complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            **kwargs: Additional parameters passed to the LiteLLM API

        Returns:
            Returns the full generated text
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        
        # Convert prompt to messages format for LiteLLM
        messages = [{"role": "user", "content": prompt}]
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            "api_base": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = litellm.completion(**params)
                
                full_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        full_text += chunk.choices[0].delta.content
                return full_text
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)
