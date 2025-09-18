import logging
import multiprocessing
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from datasets import Dataset
from google import genai
from google.genai import types
from tqdm import tqdm

# ==============================================
# utils
# ==============================================


def save_string_to_file(content, file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def read_str_from_file(filepath: str, encoding: str = "utf-8") -> str:
    try:
        with open(filepath, "r", encoding=encoding) as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise
    except IOError:
        raise


# ==============================================
# call Gemini
# ==============================================


def create_one_gemini_response(
    prompt,
    model="gemini-2.5-pro",
    tools=[],
    cache_path=None,
    use_cache=False,
    save_cache=True,
    api_key=os.getenv("GEMINI_API_KEY"),
):
    # 若 use_cache，检查是否有 cache_path 这个文件，若有，则直接读取文件并返回
    client_gemini = genai.Client(api_key=api_key)
    if use_cache and cache_path:
        if os.path.exists(cache_path):
            logging.info(f"using cache {cache_path}")
            return read_str_from_file(cache_path)

    generated_text = ""
    try:
        config = types.GenerateContentConfig(
            tools=tools
        )
        config = types.GenerateContentConfig(
            tools=tools
        )
        response = client_gemini.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        generated_text = response.text
    except Exception as e:
        print(f"Error processing prompt: {prompt[:20]}... - {e}")
        generated_text = ""
    # 若 save_cache，保存到 cache_path
    if save_cache and cache_path:
        logging.info(f"save to cache {cache_path}")
        save_string_to_file(generated_text, cache_path)
    return generated_text


def create_gemini_responses_with_pool(
    dataset,
    prompt_column_name="prompt",
    response_column_name="response",
    unique_key_name="id",
    num_workers=4,
    model="gemini-2.5-pro",
    tools=[],
    cache_dir=None,
    use_cache=False,
    save_cache=True,
):
    """
    Create Gemini responses for a dataset using multithreading with a thread pool.
    
    Args:
        dataset: Dataset object or list of dictionaries
        prompt_column_name: Name of the column containing prompts
        response_column_name: Name of the column to store responses
        unique_key_name: Name of the unique identifier column
        num_workers: Number of worker threads
        model: Gemini model to use
        tools: Tools to use
        cache_dir: Directory to store cache files
        use_cache: Whether to use cached responses
        save_cache: Whether to save responses to cache
    
    Returns:
        Dataset with responses added
    """
    # Convert to list if it's a Dataset object
    if hasattr(dataset, 'to_list'):
        data_list = dataset.to_list()
    else:
        data_list = dataset
    
    # Thread lock for safe logging
    log_lock = threading.Lock()
    
    def process_single_item(item):
        """Process a single item with thread-safe logging"""
        try:
            prompt = item[prompt_column_name]
            unique_key = item[unique_key_name]
            
            # Set up cache path if cache is enabled
            cache_path = None
            if cache_dir and (use_cache or save_cache):
                cache_path = os.path.join(cache_dir, f"{unique_key}.txt")
            
            # Generate response
            response = create_one_gemini_response(
                prompt=prompt,
                model=model,
                tools=tools,
                cache_path=cache_path,
                use_cache=use_cache,
                save_cache=save_cache,
            )
            
            # Create result item
            result_item = item.copy()
            result_item[response_column_name] = response
            
            with log_lock:
                logging.info(f"Processed item {unique_key}")
            
            return result_item
            
        except Exception as e:
            with log_lock:
                logging.error(f"Error processing item {item.get(unique_key_name, 'unknown')}: {e}")
            
            # Return item with empty response on error
            result_item = item.copy()
            result_item[response_column_name] = ""
            return result_item
    
    # Process items with thread pool
    results = []
    
    progress_bar = tqdm(total=len(data_list), desc="Processing prompts", unit="item")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_item, item): item 
            for item in data_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert back to Dataset if input was a Dataset
    if hasattr(dataset, 'to_list'):
        return Dataset.from_list(results)
    else:
        return results





if __name__ == "__main__":
    ds = [
        {"prompt": "Hi, who are you?", "id": "1"},
        {"prompt": "Calculate 1 + 1.", "id": "2"},
        {"prompt": "Tell me a joke.", "id": "3"},
        {"prompt": "Give me three day events schedule based on https://www.weather.com.cn/shanghai/index.shtml. Also let me know what needs to taken care of considering weather and commute.", "id": "4"},
    ]
    tools = [
        {"url_context": {}},
        {"google_search": {}},
    ]
    ds = Dataset.from_list(ds)
    res_ds = create_gemini_responses_with_pool(
        ds,
        prompt_column_name="prompt",
        response_column_name="response",
        unique_key_name="id",
        num_workers=10,
        tools=tools,
    )
    for row in res_ds:
        id = row["id"]
        print(f"\n-------------[{id:^5}]--------------\n")
        print(row["response"])
