import openai
import asyncio
import threading
from tqdm.asyncio import tqdm
import time
from openai.error import Timeout as APITimeoutError, RateLimitError
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import re

class APIUsageTracker(object):
    """A singleton class to track API usage."""

    _instance = None
    _token_usage_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIUsageTracker, cls).__new__(cls)
            cls._instance.token_usage = {}
        return cls._instance

    def get_token_usage(self):
        return self.token_usage
    
    def get_model_usage(self, model):
        return self.token_usage.get(model, {'prompt_tokens': 0, 'completion_tokens': 0})

    
    def reset_token_usage(self):
        self.token_usage = {}

    def increment_token_usage(self, model, prompt_tokens, completion_tokens):
        with self._token_usage_lock:
            if model in self.token_usage:
                self.token_usage[model]['prompt_tokens'] += prompt_tokens
                self.token_usage[model]['completion_tokens'] += completion_tokens
            else:
                self.token_usage[model] = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}
    
    @staticmethod
    def format_token_count(count):
        """Return a compact token count (e.g. 1.2K, 3.4M)."""
        for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if count >= threshold:
                value = count / threshold
                return f"{value:.1f}".rstrip("0").rstrip(".") + suffix
        return str(count)

api_usage_tracker = APIUsageTracker()

def record_api_usage(func):
    """Decorator to record API usage for OpenAI API calls."""
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        usage = response.get('usage', {})
        model = kwargs.get("model")
        
        prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
        completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))

        api_usage_tracker.increment_token_usage(model, prompt_tokens, completion_tokens)
        return response
    return wrapper

class OpenAIModel:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = "http://115.156.159.110:9833/"
        openai.api_version = "2025-04-01-preview"
        openai.api_key = "sk-hrsgcptaqccbqhwiutusulvlctwmkqvinxktkwrarscddqjr"

        self.max_retries = 99999
        self.initial_backoff = 5
    
    @record_api_usage
    def get_response_with_retry(self, *, input, model, method="completions", validator=None, **extra):
        """Call the model with retries and validation."""
        retry_ctr = 0
        delay = self.initial_backoff
        while retry_ctr < self.max_retries:
            if retry_ctr > 0:
                print(f"Previous Generation Failed, Retrying... {retry_ctr+1}/{self.max_retries}")
            try:
                if method == "completions":
                    response = openai.ChatCompletion.create(
                        engine=model,
                        messages=input,
                        **extra,
                    )
                elif method == "responses":
                    response = openai.Completion.create(
                        engine=model,
                        prompt=input,
                        **extra,
                    )
                response_text = self.extract_response_text(response, method=method)
                if validator is None or validator(response_text):
                    return response
                else:
                    print("Validation failed.")
            except TimeoutError:
                print(f"Timeout")
                delay *= 2
                time.sleep(delay)
            except APITimeoutError:
                print(f"APITimeout")
                delay *= 2
                time.sleep(delay)
            except RateLimitError as r:
                print(f"Rate_limit {r}")
                time.sleep(30)
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(10)
            retry_ctr += 1
        raise RuntimeError("Max retries exceeded")
    
    def extract_response_text(self, response, method="completions"):
        if method == "completions":
            return response['choices'][0]['message']['content']
        elif method == "responses":
            return response['choices'][0]['text']
    
    async def async_get_response_with_retry(self, *, input, model, validator=None, **extra):
        """Asynchronously call the model with retries and validation."""
        retry_ctr = 0
        delay = self.initial_backoff
        while retry_ctr < self.max_retries:
            if retry_ctr > 0:
                print(f"Previous Generation Failed, Retrying... {retry_ctr+1}/{self.max_retries}")
            try:
                response = await openai.Completion.acreate(
                    engine=model,
                    prompt=input,
                    **extra,
                )
                if validator is None or validator(response):
                    return response
            except APITimeoutError:
                print(f"Timeout")
                delay *= 2
                await asyncio.sleep(delay)
            except RateLimitError as r:
                print(f"Rate_limit {r}")
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Exception: {e}")
                await asyncio.sleep(10)
            retry_ctr += 1
        raise RuntimeError("Max retries exceeded")
    
    async def async_get_multi_response(self, *, inputs, model, **extra):
        """Asynchronously call the model with multiple inputs and return the responses."""
        
        tasks = [
            self.async_get_response_with_retry(input=input, model=model, **extra)
            for input in inputs
        ]
        responses = await tqdm.gather(*tasks)
        return responses
    
    def get_multi_response(self, *, inputs, model, max_workers=4, base_timeout=None, method="completions", **kwargs):
        base_timeout = base_timeout or 240 * len(inputs) / max_workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            timeout = base_timeout
        
            results = executor.map(
                lambda input: self.get_response_with_retry(input=input, model=model, method="completions", **kwargs),
                inputs,
                timeout=timeout,
            )

            for res in results:
                yield res
                        # answered_prompts.append()

class Validator:
    @staticmethod
    def json_validator(response):
        """Ensure returned JSON includes expected user keys."""
        try:
            parsed = json.loads(response)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def information_validator(response):
        check_list = ["[Contextual Emails]:", "[Contextual Chats]:", "[Request Email]:"]
        return all(check in response for check in check_list)
    
def bind_validator(func):
    """
    Decorator that attaches a .validate() method to the decorated function.
    The .validate() method returns True if the function executes without raising an exception,
    and False otherwise.
    """
    def validate(response_text):
        try:
            func(response_text)
            return True
        except Exception:
            return False
    
    func.validate = validate
    return func

class Extractor:
    @staticmethod
    def information_extractor(response):
        sections = ["[Contextual Emails]:", "[Contextual Chats]:", "[Request Email]:"]
        pattern = '|'.join([re.escape(sec) for sec in sections])
        splits = re.split(pattern, response)
        results = {}
        for i, sec in enumerate(sections):
            results[sec.strip("[]:")] = splits[i+1].strip()
        return results