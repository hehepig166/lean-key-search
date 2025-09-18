from datasets import load_dataset, Dataset


def match_query(query: str, row: dict):
    """
    query:  +xx -xx +yy -yy +zz -zz tt #dd ...
            xx, yy, zz, tt, dd are the keys to match,

            +xx or xx means the key must be in the lean_name or docstring
            -xx means the key must not be in the lean_name,
            #dd means the key must be in the docstring, (if the docstring is empty, it will be ignored)
            @dd means the key must be in the docstring, (if the docstring is empty, return False)

            the lean_name is row["lean_name"]
            the docstring is row["docstring"]
            


    return: True if the query is matched, False otherwise
    """
    keys = query.split(" ")
    for key in keys:
        key = key.lower()
        name = row.get("lean_name", "")
        docstring = row.get("docstring", "")
        if not name:
            name = ""
        if not docstring:
            docstring = ""
        name = name.lower()
        docstring = docstring.lower()

        if key.startswith("+"):
            key = key[1:]
            if key in docstring:
                continue
            if key in name:
                continue
            return False
        elif key.startswith("-"):
            key = key[1:]
            if key in name:
                return False
        elif key.startswith("#"):
            key = key[1:]
            if docstring is None or docstring == "":
                return True
            if key not in docstring:
                return False
        elif key.startswith("@"):
            key = key[1:]
            if key not in docstring:
                return False
        else:
            if key not in name:
                return False
    return True
        


def interactive_search(dataset: Dataset):
    showing_limit = 3
    columns_to_show = ["id", "lean_name", "declaration_signature", "docstring"]
    while True:
        query = input("Enter your search query: ")
        if query.strip() == "":
            continue
        results = dataset.filter(lambda x: match_query(query, x), num_proc=8)
        print(f"Found {len(results)} results\n\n")

        for i in range(min(showing_limit, len(results))):
            for column in columns_to_show:
                print(f"[{column}]\n{results[i][column]}\n------")
            print()
            print("=" * 30)
            print()
        

def search_tool(dataset: Dataset, query: str, page: int = 1, items_per_page: int = 10):
    """
    Search tool that returns paginated matching results for a given query.
    
    Args:
        dataset: The dataset to search in
        query: Search query string (supports +xx -xx #xx @xx syntax)
        page: Page number (1-based, default: 1)
        items_per_page: Number of items per page (default: 10)
    
    Returns:
        Dictionary with paginated results and metadata
    """
    results = dataset.filter(lambda x: match_query(query, x), num_proc=8)
    total_matches = len(results)
    
    # Calculate pagination
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    # Convert to list and apply pagination
    results_list = []
    for i in range(start_idx, min(end_idx, total_matches)):
        row = results[i]
        # Extract relevant information for the agent
        result_info = {
            "id": row.get("id"),
            "lean_name": row.get("lean_name"),
            "decl_type": row.get("decl_type"),
            "declaration_signature": row.get("declaration_signature"),
            "docstring": row.get("docstring"),
            "statement_text": row.get("statement_text"),
            "module_name": row.get("module_name")
        }
        results_list.append(result_info)
    
    # Calculate pagination metadata
    total_pages = (total_matches + items_per_page - 1) // items_per_page
    has_next = page < total_pages
    has_prev = page > 1
    
    return {
        "total_matches": total_matches,
        "total_pages": total_pages,
        "current_page": page,
        "items_per_page": items_per_page,
        "returned_results": len(results_list),
        "has_next": has_next,
        "has_prev": has_prev,
        "results": results_list
    }


def select_useful_rows(dataset: Dataset):
    useful_keys = [
        "Nat", "Int", "Real", "Finset", "Set",
        "sum", "prod",
        "prime", "dvd", "factor",
        "sin", "cos", "tan", "log", "gcd", "lcm",
        "chinese_remainder", "floor", "ceil", "abs",
    ]

    selected_dataset = dataset
    # selected_dataset = selected_dataset.filter(lambda x: any(key.lower() in x["lean_name"].lower() for key in useful_keys), num_proc=8)
    selected_dataset = selected_dataset.filter(lambda x: x["declaration_signature"] is not None, num_proc=8)
    return selected_dataset

def save_selected_dataset(selected_dataset: Dataset, filename: str = "selected_mathlib_dataset.jsonl"):
    """Save the selected dataset to a local JSONL file."""
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        for item in selected_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Selected dataset saved to {filename} with {len(selected_dataset)} items")

def load_selected_dataset(filename: str = "selected_mathlib_dataset.jsonl"):
    """Load the selected dataset from a local JSONL file."""
    import json
    from datasets import Dataset
    
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return Dataset.from_list(data)

if __name__ == "__main__":
    import os
    
    selected_dataset_file = "selected_mathlib_dataset.jsonl"
    # selected_dataset_file = "declarations_dataset.jsonl"
    
    if os.path.exists(selected_dataset_file):
        print("Loading selected dataset from local file...")
        selected_dataset = load_selected_dataset(selected_dataset_file)
        print(f"Loaded {len(selected_dataset)} items from {selected_dataset_file}")
    else:
        print("Loading full dataset...")
        dataset = load_dataset("AI-MO/mathlib-declarations", split="train", num_proc=4)
        print("Selecting useful rows...")
        selected_dataset = select_useful_rows(dataset)
        print(f"Selected {len(selected_dataset)} useful items")
        save_selected_dataset(selected_dataset, selected_dataset_file)
    
    interactive_search(selected_dataset)
