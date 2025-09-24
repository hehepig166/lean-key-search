"""
Mathlib Search Agent - A modular search agent for Mathlib theorem search.

This module provides a flexible architecture for building search agents that can
interact with Mathlib theorem databases using different response providers.
"""

import os
import re
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests
from search_dataset import load_selected_dataset, search_tool
from llm_utils import create_one_gemini_response
from openai import OpenAI


class ResponseProvider(ABC):
    """Abstract base class for response providers."""
    
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the provider.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            The response string
        """
        pass
    
    @abstractmethod
    def clean_response(self, response: str) -> str:
        """
        Clean the response (e.g., remove thinking tags).
        
        Args:
            response: Raw response string
            
        Returns:
            Cleaned response string
        """
        pass


class StdinResponseProvider(ResponseProvider):
    """Response provider that gets responses from stdin (for manual testing)."""
    
    def get_response(self, prompt: str) -> str:
        """Get response from stdin with prompt display."""
        # Display prompt
        print("\n📤 PROMPT sent to LLM:")
        print("="*60)
        print(prompt)
        print("="*60)
        
        # Save to __prompt.txt
        try:
            with open("__prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            print("💾 Prompt saved to __prompt.txt")
        except Exception as e:
            print(f"❌ Failed to save prompt: {e}")
        
        print("\n📥 Please enter agent response:")
        print("(Can include <think> tags, will be automatically cleaned)")
        print("Type 'END' to finish response input")
        print("-" * 40)
        
        response_lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            response_lines.append(line)
        
        return '\n'.join(response_lines)
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing thinking tags."""
        return self._clean_thinking_tags(response)
    
    def _clean_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from text."""
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        return cleaned


class OllamaResponseProvider(ResponseProvider):
    """Response provider that gets responses from local Ollama."""
    
    def __init__(self, model: str = "deepseek-r1", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama response provider.
        
        Args:
            model: Ollama model name
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
    
    def get_response(self, prompt: str) -> str:
        """Get response from Ollama."""
        try:
            # Convert prompt string back to messages format for Ollama
            messages = self._parse_prompt_to_messages(prompt)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": 2048,
                        "temperature": 0.1
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
    
    def _parse_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Parse formatted prompt string back to messages format."""
        messages = []
        lines = prompt.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            if line.endswith(':') and line[:-1].lower() in ['system', 'user', 'assistant', 'tool']:
                # Save previous message if exists
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new message
                current_role = line[:-1].lower()
                current_content = []
            else:
                if current_role:
                    current_content.append(line)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })
        
        return messages
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing thinking tags."""
        return self._clean_thinking_tags(response)
    
    def _clean_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from text."""
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        return cleaned


class GeminiResponseProvider(ResponseProvider):
    """Response provider that gets responses from Google Gemini."""
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini response provider.
        
        Args:
            model: Gemini model name
            api_key: Google API key (if None, will use GEMINI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
    
    def get_response(self, prompt: str) -> str:
        """Get response from Gemini."""
        try:
            response = create_one_gemini_response(
                prompt=prompt,
                model=self.model,
                api_key=self.api_key
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Gemini: {e}")
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing thinking tags."""
        return self._clean_thinking_tags(response)
    
    def _clean_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from text."""
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        return cleaned


class OpenAIResponseProvider(ResponseProvider):
    """Response provider that gets responses from OpenAI."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI response provider.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (if None, will use default OpenAI endpoint)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def get_response(self, prompt: str) -> str:
        """Get response from OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to get response from OpenAI: {e}")
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing thinking tags."""
        return self._clean_thinking_tags(response)
    
    def _clean_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from text."""
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        return cleaned


class MathlibSearchAgent:
    """Main search agent for Mathlib theorem search."""
    
    def __init__(self, 
                 response_provider: ResponseProvider,
                 max_searches: int = 15,
                 prompts_dir: str = "prompts_log"):
        """
        Initialize the Mathlib search agent.
        
        Args:
            response_provider: Provider for getting responses
            max_searches: Maximum number of searches per query
            prompts_dir: Directory to save prompts
        """
        self.response_provider = response_provider
        self.max_searches = max_searches
        self.prompts_dir = prompts_dir
        self.conversation_history = []
        self.search_count = 0
        
        # Create prompts directory
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            with open("mathlib_agent_prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print("❌ Error: mathlib_agent_prompt.txt not found!")
            print("Please ensure the system prompt file exists in the current directory.")
            print("You can copy it from old_files/ directory if needed.")
            exit(1)
    
    def _load_dataset(self):
        """Load the Mathlib dataset."""
        try:
            print("Loading Mathlib dataset...")
            dataset = load_selected_dataset()
            print(f"✅ Dataset loaded with {len(dataset)} items")
            return dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("Please ensure the dataset files are available.")
            exit(1)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def extract_search_instructions(self, text: str) -> List[str]:
        """Extract search instructions from text."""
        # Look for lines starting with "Search:"
        search_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('Search:'):
                search_lines.append(line[7:].strip())  # Remove "Search:" prefix
        return search_lines
    
    def parse_search_query(self, instruction: str) -> Tuple[str, int]:
        """Parse search instruction to extract query and page."""
        # Default to page 1 if not specified
        page = 1
        
        # Look for page specification like "query page 2", "query p2", or "query (page 2)"
        page_match = re.search(r'(?:\(page\s*(\d+)\)|\(p\s*(\d+)\)|\b(?:page|p)\s*(\d+)\b)', instruction, re.IGNORECASE)
        if page_match:
            # Extract page number from any of the matched groups
            page = int(page_match.group(1) or page_match.group(2) or page_match.group(3))
            # Remove page specification from query (handle both formats)
            instruction = re.sub(r'(?:\(page\s*\d+\)|\(p\s*\d+\)|\b(?:page|p)\s*\d+\b)', '', instruction, flags=re.IGNORECASE).strip()
        
        return instruction, page
    
    def execute_search(self, query: str, page: int = 1) -> str:
        """Execute a search using the search tool."""
        try:
            result = search_tool(self.dataset, query, page=page, items_per_page=5)
            return self._format_search_result(result)
        except ImportError:
            return f"Search tool not available. Query: {query}, Page: {page}"
        except Exception as e:
            return f"Search error: {e}"
    
    def _format_search_result(self, result: Dict[str, Any]) -> str:
        """Format search result for display."""
        if "error" in result:
            return f"Search error: {result['error']}"
        
        # Check if page is out of range
        if result['current_page'] > result['total_pages'] and result['total_pages'] > 0:
            return f"Page {result['current_page']} is out of range. Only {result['total_pages']} pages available (total matches: {result['total_matches']})."
        
        if not result["results"]:
            if result['total_matches'] == 0:
                return f"No results found for query."
            else:
                return f"No results found for query. Total matches: {result['total_matches']}"
        
        formatted = f"Found {result['returned_results']} results (page {result['current_page']}/{result['total_pages']}, total matches: {result['total_matches']}):\n\n"
        
        for i, theorem in enumerate(result["results"], 1):  # Show all results
            formatted += f"**{i}. {theorem['lean_name']}**\n"
            formatted += f"   Type: {theorem['decl_type']}\n"
            formatted += f"   Statement: `{theorem['declaration_signature']}`\n"
            if theorem.get('docstring'):
                formatted += f"   Docstring: {theorem['docstring'][:100]}...\n"
            formatted += f"   Module: {theorem['module_name']}\n\n"
        
        return formatted
    
    def _format_messages_for_display(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for display."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Format role names properly
            if role == "system":
                # Add search budget info to system prompt
                remaining_searches = self.max_searches - self.search_count
                budget_info = f"\n\n[Search Budget: {remaining_searches}/{self.max_searches} searches remaining]"
                content = content + budget_info
                formatted.append(f"System:\n{content}")
            elif role == "user":
                formatted.append(f"User:\n{content}")
            elif role == "assistant":
                formatted.append(f"Assistant:\n{content}")
            elif role == "tool":
                formatted.append(f"Tool:\n{content}")
            else:
                formatted.append(f"{role.title()}:\n{content}")
        
        return "\n\n".join(formatted)
    
    def _save_prompt_to_file(self, prompt: str, user_query: str = "") -> None:
        """Save prompt to file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_prompt_{timestamp}.txt"
        filepath = os.path.join(self.prompts_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if user_query:
                    f.write(f"Query: {user_query}\n")
                    f.write("="*60 + "\n\n")
                f.write(prompt)
            print(f"✅ Prompt saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save prompt: {e}")
    
    def _save_final_result(self, user_query: str, final_response: str) -> None:
        """Save final result to __result.txt file and display it."""
        try:            
            # Save to __result.txt
            with open("__result.txt", "w", encoding="utf-8") as f:
                f.write(final_response)
            print(f"💾 Final result saved to: __result.txt")
            
            # Check if any theorems were found in the conversation
            has_found_theorems = self._check_if_theorems_found()
            
            if has_found_theorems:
                print(f"\n✅ Successfully found relevant theorems!")
            else:
                print(f"\n❌ No relevant theorems found")
            
            # Display final response
            print(f"\n📄 Final Response:")
            print("="*60)
            print(final_response)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Failed to save final result: {e}")
    
    def _check_if_theorems_found(self) -> bool:
        """Check if any theorems were found by looking for Cover match in assistant responses."""
        # Look through conversation history for assistant messages with Cover match
        for msg in self.conversation_history:
            if msg["role"] == "assistant" and "**Cover match**:" in msg["content"]:
                content = msg["content"]
                # Check if Cover match is not None
                if "**Cover match**: None" not in content and "**Cover match**: " in content:
                    return True
        return False
    
    def handle_query(self, user_query: str) -> None:
        """Handle a single user query with multi-turn interaction."""
        print(f"\n🔍 Processing query: {user_query}")
        print("="*60)
        
        # Reset conversation history and search count for new query
        self.conversation_history = []
        self.search_count = 0
        
        # Add user message
        self.add_message("user", user_query)
        
        # Multi-turn interaction loop
        while True:
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            
            # Format messages for provider
            formatted_prompt = self._format_messages_for_display(messages)
            
            # Save prompt to file
            self._save_prompt_to_file(formatted_prompt, user_query)
            
            # Get response from provider
            raw_response = self.response_provider.get_response(formatted_prompt)
            cleaned_response = self.response_provider.clean_response(raw_response)
            
            print("\n🧹 Cleaned response:")
            print("="*60)
            print(cleaned_response)
            print("="*60)
            
            # Check for search instructions
            search_instructions = self.extract_search_instructions(cleaned_response)
            
            if search_instructions and self.search_count < self.max_searches:
                print(f"\n🔍 Detected search instructions: {search_instructions}")
                
                # Execute searches
                search_results = []
                for instruction in search_instructions:  # Execute all search instructions
                    if self.search_count >= self.max_searches:
                        break
                    
                    query, page = self.parse_search_query(instruction)
                    print(f"\n🔍 Executing search: '{query}' (page {page})")
                    
                    result = self.execute_search(query, page)
                    search_results.append(f"Search: {instruction}\n{result}")
                    
                    self.search_count += 1
                    print(f"✅ Search {self.search_count}/{self.max_searches} completed")
                
                # Add results to conversation
                if search_results:
                    search_content = "\n\n".join(search_results)
                    self.add_message("assistant", cleaned_response)
                    self.add_message("tool", f"Search results:\n{search_content}")
                    
                    # Continue for next turn
                    continue
            else:
                # No search instructions - final answer
                print("\n✅ Agent provided final answer (no search instructions)")
                self.add_message("assistant", cleaned_response)
                
                # Save final result
                self._save_final_result(user_query, cleaned_response)
                break
        
        print(f"\n[Search statistics: {self.search_count}/{self.max_searches} used]")
        print("-" * 50)
    
    def reset_conversation(self) -> None:
        """Reset conversation history and search count."""
        self.conversation_history = []
        self.search_count = 0
        print("✅ Conversation reset")
    
    def run_interactive_session(self) -> None:
        """Run interactive session with the agent."""
        print("🤖 Mathlib Search Agent - Interactive Session")
        print("="*60)
        print("📝 Note: Conversation history is preserved, search budget accumulates across session")
        print("   Use 'reset' command to manually clear history and reset search budget")
        print("")
        print("Type 'quit' to exit, 'reset' to manually reset conversation")
        print("="*60)
        
        while True:
            print("\n📝 Please enter your query:")
            print("(Type 'END' to finish input, 'quit' to exit, 'reset' to reset conversation)")
            print("-" * 40)
            
            user_input_lines = []
            while True:
                line = input()
                if line.strip().lower() in ['quit', 'exit', 'q']:
                    return  # Exit the entire session
                elif line.strip().lower() == 'reset':
                    self.reset_conversation()
                    break  # Break inner loop to get new input
                elif line.strip() == 'END':
                    break  # Finish input
                else:
                    user_input_lines.append(line)
            
            # Check if we should continue (reset command)
            if not user_input_lines:
                continue
            
            user_input = '\n'.join(user_input_lines).strip()
            if not user_input:
                continue
            
            self.handle_query(user_input)


def main():
    """Main function to run the agent."""
    # Choose response provider
    provider_type = input("Choose response provider (1: stdin, 2: ollama, 3: gemini, 4: openai): ").strip()
    
    if provider_type == "2":
        model = input("Enter Ollama model name (default: deepseek-r1): ").strip() or "deepseek-r1"
        provider = OllamaResponseProvider(model=model)
    elif provider_type == "3":
        model = input("Enter Gemini model name (default: gemini-2.5-flash): ").strip() or "gemini-2.5-flash"
        api_key = input("Enter Gemini API key (or press Enter to use GEMINI_API_KEY env var): ").strip() or None
        provider = GeminiResponseProvider(model=model, api_key=api_key)
    elif provider_type == "4":
        model = input("Enter OpenAI model name (default: gpt-4o): ").strip() or "gpt-4o"
        api_key = input("Enter OpenAI API key (or press Enter to use OPENAI_API_KEY env var): ").strip() or None
        base_url = input("Enter OpenAI base URL (or press Enter for default): ").strip() or None
        provider = OpenAIResponseProvider(model=model, api_key=api_key, base_url=base_url)
    else:
        provider = StdinResponseProvider()
    
    # Create and run agent
    agent = MathlibSearchAgent(response_provider=provider)
    agent.run_interactive_session()


if __name__ == "__main__":
    main()
