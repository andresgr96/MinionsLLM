"""LLM-based behavior tree generation using various prompting techniques."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

# Optional imports - graceful failure if not available
try:
    from llama_cpp import Llama

    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from dotenv import load_dotenv  # type: ignore
from lancedb import table
from lancedb.pydantic import LanceModel
from openai import OpenAI
from vi import Agent


class BehaviorTreeGenerator:
    """Generator for behavior trees using LLM-based prompting techniques."""

    def __init__(
        self,
        agent_class: Type[Agent],
        backend: str = "llamacpp",
        model_path_or_url: Optional[str] = None,
        ollama_model_name: Optional[str] = None,
        chat_format: str = "llama-3",
        context_length: int = 4096,
        gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 150,
        repeat_penalty: float = 1.2,
        seed: int = 100,
        client: Optional[OpenAI] = None,
        vector_tb: Optional[table.Table] = None,
        schema: Optional[Type[LanceModel]] = None,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BehaviorTreeGenerator with support for multiple backends.

        Args:
            agent_class: The agent class to generate behavior trees for
            backend: Backend to use ("llamacpp" or "ollama", default: "llamacpp")
            model_path_or_url: Path to .gguf file (for llamacpp/Ollama) or HuggingFace URL (for Ollama)
            ollama_model_name: Existing Ollama model name
            chat_format: Chat format for the model (default: "llama-3")
            context_length: Context length for the model (default: 4096)
            gpu_layers: Number of GPU layers to use (llamacpp only, default: -1)
            n_threads: Number of threads to use (llamacpp only, default: None)
            temperature: Temperature for sampling (default: 0.0)
            top_p: Top-p sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 150)
            repeat_penalty: Penalty for repeating tokens (default: 1.2)
            seed: Random seed (default: 100)
            client: OpenAI client for embeddings (optional, only needed for RAG)
            vector_tb: LanceDB table for storing examples (optional, only needed for RAG)
            schema: LanceDB schema for the table (optional, only needed for RAG)
            max_retries: Maximum number of retries for behavior tree generation (default: 1)
            **kwargs: Additional backend-specific parameters
        """
        load_dotenv()

        # Validate backend
        if backend not in ["llamacpp", "ollama"]:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be 'llamacpp' or 'ollama'"
            )

        self.backend = backend
        self.agent = agent_class
        self.error_count = 0
        self.max_retries = max_retries

        # Common parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.context_length = context_length
        self.chat_format = chat_format

        # RAG setup
        self.vector_tb = vector_tb
        self.client = client
        self.schema = schema
        self.rag_enabled = all([self.client, self.vector_tb, self.schema])

        # Backend-specific initialization
        if backend == "llamacpp":
            self._init_llamacpp(model_path_or_url, gpu_layers, n_threads, **kwargs)
        elif backend == "ollama":
            self._init_ollama(model_path_or_url, ollama_model_name, **kwargs)

    def _validate_llamacpp_requirements(self, model_path: str) -> None:
        """Validate requirements for llamacpp backend."""
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for llamacpp backend")

        if not model_path:
            raise ValueError("model_path_or_url is required for llamacpp backend")

        if not model_path.endswith(".gguf"):
            raise ValueError("LlamaCPP requires a .gguf model file")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _validate_ollama_requirements(
        self, model_path_or_url: Optional[str], ollama_model_name: Optional[str]
    ) -> None:
        """Validate requirements for ollama backend."""
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama is required for ollama backend. Install with: pip install ollama (also needs to be downloaded and installed from https://ollama.com)"
            )

        if not model_path_or_url and not ollama_model_name:
            raise ValueError(
                "Either model_path_or_url or ollama_model_name is required for ollama backend"
            )

        # Check if Ollama service is running
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Ollama service is not running or accessible. Error: {e}"
            )

    def _init_llamacpp(
        self,
        model_path_or_url: Optional[str],
        gpu_layers: int,
        n_threads: Optional[int],
        **kwargs: Any,
    ) -> None:
        """Initialize llamacpp backend."""
        if model_path_or_url is None:
            raise ValueError("model_path_or_url is required for llamacpp backend")
        self._validate_llamacpp_requirements(model_path_or_url)

        self.model_path = model_path_or_url
        self.gpu_layers = gpu_layers
        self.n_threads = n_threads if n_threads is not None else os.cpu_count()

        # Filter kwargs for llamacpp
        llamacpp_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["echo", "logits_all", "vocab_only", "use_mmap", "use_mlock"]
        }

        print(f"Initializing llamacpp with model: {model_path_or_url}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=self.n_threads,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            seed=self.seed,
            chat_format=self.chat_format,
            verbose=False,
            **llamacpp_kwargs,
        )

    def _init_ollama(
        self,
        model_path_or_url: Optional[str],
        ollama_model_name: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Initialize ollama backend."""
        self._validate_ollama_requirements(model_path_or_url, ollama_model_name)

        # Determine the model to use
        if ollama_model_name:
            # Use existing Ollama model
            self.ollama_model = ollama_model_name
            print(f"Using existing Ollama model: {ollama_model_name}")
        elif model_path_or_url:
            if model_path_or_url.endswith(".gguf"):
                # Auto-import local GGUF file
                self.ollama_model = self._auto_import_gguf(model_path_or_url)
            elif (
                "hf.co/" in model_path_or_url or "huggingface.co/" in model_path_or_url
            ):
                # Pull from HuggingFace
                print(f"Pulling model from HuggingFace: {model_path_or_url}")
                self.ollama_model = self._pull_model(model_path_or_url)
            else:
                raise ValueError(
                    "model_path_or_url must be either a .gguf file path or a HuggingFace URL"
                )

        # Verify the model exists in Ollama
        self._verify_ollama_model(self.ollama_model)
        print(f"Ollama backend initialized with model: {self.ollama_model}")

    def _pull_model(self, model_name: str) -> str:
        """Pull a model using Ollama (supports both Ollama models and HuggingFace URLs)."""
        try:
            print(f"Pulling model '{model_name}'...")
            print("This may take a while depending on model size...")

            ollama.pull(model_name)

            # List models to show the user
            models = ollama.list()
            print("\nAvailable models:")
            print(models)

            print(f"Model '{model_name}' pulled successfully!")
            return model_name

        except Exception as e:
            error_str = str(e)
            if "401" in error_str and "Invalid username or password" in error_str:
                print("\n" + "=" * 60)
                print("GATED MODEL DETECTED")
                print("=" * 60)
                print(
                    "You are trying to access a gated/private model from HuggingFace. First make sure you have access to the model."
                )
                print(
                    "To access gated models with Ollama, you need to set up SSH key authentication."
                )
                print("\nFollow these steps:")
                print("\n1. Get your Ollama SSH key:")
                print(
                    "   Windows: Get-Content $env:USERPROFILE\\.ollama\\id_ed25519.pub"
                )
                print("   Linux/Mac: cat ~/.ollama/id_ed25519.pub")
                print("\n2. Copy the SSH key output")
                print("\n3. Add it to your HuggingFace account:")
                print("   - Go to: https://huggingface.co/settings/keys")
                print("   - Click 'Add new SSH key'")
                print("   - Paste your SSH key and save")
                print("\n4. Make sure you have access to the gated model:")
                print("   - Visit the model page and request access if needed")
                print("\nFor detailed instructions, see:")
                print(
                    "https://huggingface.co/docs/hub/ollama#run-private-ggufs-from-the-hugging-face-hub"
                )
                print("=" * 60)

                raise RuntimeError(
                    "SSH key authentication required for gated model. Please follow the instructions above."
                )

            # For non-authentication errors
            self._suggest_similar_models(model_name)
            raise RuntimeError(f"Failed to pull model '{model_name}': {e}")

    def _auto_import_gguf(self, gguf_path: str) -> str:
        """Auto-import a GGUF file into Ollama and return the model name."""
        gguf_path = os.path.abspath(gguf_path)
        model_name = f"custom-{Path(gguf_path).stem}"

        print(f"Auto-importing GGUF file: {gguf_path}")
        print(f"Creating Ollama model: {model_name}")

        # Create a temporary Modelfile
        modelfile_content = f"FROM {gguf_path}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        try:
            # print(f"User Name: {os.getenv('USERNAME')}")
            # Try to find ollama executable in common Windows locations
            possible_ollama_paths = [
                "ollama",
                "ollama.exe",
                r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(
                    os.getenv("USERNAME", "")
                ),
            ]

            ollama_cmd = None
            for cmd in possible_ollama_paths:
                try:
                    result = subprocess.run(
                        [cmd, "--version"], capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        ollama_cmd = cmd
                        print(f"Found Ollama at: {cmd}")
                        break
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    continue

            if not ollama_cmd:
                raise FileNotFoundError("Ollama CLI not found")

            # Create the model using ollama create
            result = subprocess.run(
                [ollama_cmd, "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"Successfully imported model as: {model_name}")
            # Return with :latest suffix since that's how Ollama stores it
            return f"{model_name}:latest"

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to import GGUF file: {e.stderr}")
        except FileNotFoundError:
            # Try alternative: use ollama.create() if CLI fails
            try:
                print("CLI not found, trying ollama.create() method...")
                with open(modelfile_path, "r") as f:
                    modelfile_content = f.read()

                ollama.create(model=model_name, modelfile=modelfile_content)
                print(f"Successfully imported model as: {model_name}")
                # Return with :latest suffix since that's how Ollama stores it
                return f"{model_name}:latest"
            except Exception as create_error:
                # If API also fails, try one more approach - stream the creation
                try:
                    print("Trying streaming create...")
                    for response in ollama.create(
                        model=model_name, modelfile=modelfile_content, stream=True
                    ):
                        if "status" in response:
                            print(f"Status: {response['status']}")
                    print(f"Successfully imported model as: {model_name}")
                    # Return with :latest suffix since that's how Ollama stores it
                    return f"{model_name}:latest"
                except Exception as stream_error:
                    raise RuntimeError(
                        f"Failed to import GGUF file. CLI error: ollama command not found. API error: {create_error}. Stream error: {stream_error}"
                    )
        finally:
            # Clean up temporary file
            os.unlink(modelfile_path)

    def _verify_ollama_model(self, model_name: str) -> None:
        """Verify that the model exists in Ollama, download if necessary."""
        # Normalize the model name - add :latest if not present
        if ":" not in model_name:
            search_model_name = f"{model_name}:latest"
        else:
            search_model_name = model_name

        try:
            # First check if Ollama service is accessible
            try:
                models_response = ollama.list()
                # print(f"Debug: Ollama list response: {models_response}")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama service. Make sure Ollama is running. Error: {e}"
                )

            # Handle Ollama ListResponse object
            available_models = []
            if hasattr(models_response, "models"):
                # This is an ollama._types.ListResponse object
                for model_obj in models_response.models:
                    if hasattr(model_obj, "model"):
                        available_models.append(model_obj.model)
                    elif hasattr(model_obj, "name"):
                        available_models.append(model_obj.name)
            elif isinstance(models_response, dict) and "models" in models_response:
                # Fallback for dict response
                models_list = models_response["models"]
                for model in models_list:
                    if isinstance(model, dict):
                        if "name" in model:
                            available_models.append(model["name"])
                        elif "model" in model:
                            available_models.append(model["model"])
                    elif isinstance(model, str):
                        available_models.append(model)
            elif isinstance(models_response, list):
                # Fallback for list response
                for model in models_response:
                    if isinstance(model, dict):
                        if "name" in model:
                            available_models.append(model["name"])
                        elif "model" in model:
                            available_models.append(model["model"])
                    elif isinstance(model, str):
                        available_models.append(model)

            # print(f"Available models in Ollama: {available_models}")

            if search_model_name not in available_models:
                # Only try to download if it doesn't start with "custom-" (local import)
                if not model_name.startswith("custom-"):
                    print(
                        f"Model '{search_model_name}' not found in Ollama. Attempting to download..."
                    )
                    self._pull_model(search_model_name)
                else:
                    raise ValueError(
                        f"Custom model '{search_model_name}' not found. It may need to be reimported."
                    )
            else:
                print(f"Model '{search_model_name}' found in Ollama")

        except Exception as e:
            if "Cannot connect to Ollama" in str(e) or "Failed to download" in str(e):
                raise e
            else:
                raise RuntimeError(f"Failed to verify Ollama model: {e}")

    def _generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate completion using the configured backend."""
        if self.backend == "llamacpp":
            return self.llm.create_chat_completion(messages=messages)  # type: ignore

        elif self.backend == "ollama":
            # Extract system and user messages for generate API
            system_content = ""
            user_content = ""

            for message in messages:
                if message["role"] == "system":
                    system_content = message["content"]
                elif message["role"] == "user":
                    user_content = message["content"]

            # Use generate API instead of chat API for single prompt/response
            response = ollama.generate(
                model=self.ollama_model,
                prompt=user_content,
                system=system_content,
                raw=False,  # Still apply chat template formatting
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repeat_penalty": self.repeat_penalty,
                    "seed": self.seed,
                    "num_ctx": self.context_length,
                },
            )

            # Convert generate response to chat format for compatibility
            return {"choices": [{"message": {"content": response["response"]}}]}

        # This should never be reached, but added for type safety
        raise ValueError(f"Unknown backend: {self.backend}")

    def query_vector_db(self, query: str) -> List[Any]:
        """
        Query the vector database for similar examples.

        Args:
            query: The query string

        Returns:
            List of similar examples

        Raises:
            RuntimeError: If RAG is not enabled or there's an error querying the database
        """
        if not self.rag_enabled:
            raise RuntimeError(
                "RAG is not enabled. Please provide client, vector_tb, and schema."
            )

        # Type assertions for mypy since we know these are not None due to rag_enabled check
        assert self.client is not None, "Client should not be None when RAG is enabled"
        assert (
            self.vector_tb is not None
        ), "Vector table should not be None when RAG is enabled"
        assert self.schema is not None, "Schema should not be None when RAG is enabled"

        try:
            response = self.client.embeddings.create(
                input=query, model="text-embedding-ada-002"
            )

            embedded_query = response.data[0].embedding
            results = (
                self.vector_tb.search(embedded_query).limit(2).to_pydantic(self.schema)
            )

            if not results:
                raise ValueError("No examples retrieved from vector database")

            return results  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Error querying vector database: {e}")

    def load_xml_template(self, file_name: str) -> str:
        """
        Load an XML template from the prompt_techniques directory.

        Args:
            file_name: The name of the template file

        Returns:
            The contents of the template file
        """
        directory = "llm_layer/prompt_techniques"
        file_path = os.path.join(directory, file_name)
        with open(file_path, "r") as file:
            return file.read()

    def generate_behavior_tree(
        self,
        prompt: str,
        fix_mistake: str = "",
        which_prompt: int = 2,
        log_prompt: bool = False,
    ) -> Tuple[int, str]:
        """
        Generate a behavior tree from a natural language prompt.

        Args:
            prompt: The natural language prompt
            fix_mistake: Error message to include in the prompt (default: "")
            which_prompt: Which prompt technique to use (default: 2, zero-shot)
                1: Zero-shot learning
                2: One-shot learning
                3: Two-shot learning
                4: RAG learning
            log_prompt: Whether to log the prompt and response (default: False)

        Returns:
            Tuple of (error_count, behavior_tree)

        Raises:
            ValueError: If an invalid prompt type is selected
            RuntimeError: If RAG is selected but not enabled
        """
        template_files = {
            1: "zero_shot_learning.xml",
            2: "one_shot_learning.xml",
            3: "two_shot_learning.xml",
            4: "rag_learning.xml",
        }

        # Check if RAG is selected but not enabled
        if which_prompt == 5 and not self.rag_enabled:
            raise RuntimeError(
                "RAG prompt selected but RAG is not enabled. Please provide client, vector_tb, and schema."
            )

        template_file = template_files.get(which_prompt)
        if not template_file:
            raise ValueError("Invalid prompt type selected.")

        # Load and prepare system prompt
        system_prompt = self.load_xml_template("system_prompt.xml")
        bt_3_example = self.load_xml_template("bt_3.xml")
        behaviors = self.call_behaviors()

        # Replace placeholders in system prompt
        system_prompt = system_prompt.replace("{bt_3.xml}", bt_3_example)
        system_prompt = system_prompt.replace("{BEHAVIORS}", str(behaviors))

        # Load and prepare n-shot learning template
        template = self.load_xml_template(template_file)

        # If using n-shot learning, load example BTs
        if which_prompt in [2, 3]:
            bt_files = {
                "BT_1": self.load_xml_template("bt_1.xml"),
                "BT_2": self.load_xml_template("bt_2.xml"),
                "BT_3": self.load_xml_template("bt_3.xml"),
                "BT_4": self.load_xml_template("bt_4.xml"),
                "BT_5": self.load_xml_template("bt_5.xml"),
            }
            # Replace BT placeholders in template
            for key, value in bt_files.items():
                template = template.replace(f"{{{key}}}", value)

        # Replace the prompt placeholder
        template = template.replace("{PROMPT}", prompt)
        template = template.replace("{FIX_MISTAKE}", fix_mistake if fix_mistake else "")

        # If using RAG, query the vector database and replace placeholders
        if which_prompt == 4:
            try:
                results = self.query_vector_db(prompt)
                template = template.replace("{RAG_PROMPT_1}", results[0].prompt)
                template = template.replace("{RAG_BT_1}", results[0].tree)
                if len(results) > 1:
                    template = template.replace("{RAG_PROMPT_2}", results[1].prompt)
                    template = template.replace("{RAG_BT_2}", results[1].tree)
            except Exception as e:
                print(f"Error using RAG: {e}")
                # Fall back to zero-shot if RAG fails
                template = self.load_xml_template("zero_shot_learning.xml")
                template = template.replace("{PROMPT}", prompt)
                template = template.replace(
                    "{FIX_MISTAKE}", fix_mistake if fix_mistake else ""
                )

        if log_prompt:
            print("###### system prompt #######")
            print(system_prompt)
            print("###### user prompt #######")
            print(template)

        # Generate the behavior tree using unified completion method
        response = self._generate_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": template},
            ]
        )

        response_text = response["choices"][0]["message"]["content"]

        if log_prompt:
            print("###### response #######")
            print(response_text)

        behavior_tree = self.extract_behavior_tree(response_text)

        if behavior_tree == "Could not generate proper XML.":
            return self.error_count, behavior_tree
        elif behavior_tree == "try again":
            fix_mistake = f"""RESPONSE: Here is your previous response: "{response_text}" it is incorrect.
            Ensure that it adheres to the correct structure, starting with the "<BehaviorTree>" root tag and ending with the "</BehaviorTree>" closing root tag and that
            you ONLY output the XML behavior tree without extra text or explanations of the tree
            USER REQUEST: Generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text or explanations of the tree."""
            return self.generate_behavior_tree(prompt, fix_mistake, which_prompt)
        else:
            return self.error_count, behavior_tree

    def call_behaviors(self) -> Dict[str, str]:
        """
        Get all the behaviors from the agent class, extracting only Node Type and Description.

        Returns:
            Dictionary of function names and their processed docstrings (Node Type and Description only)
        """
        class_obj = self.agent
        function_names_and_docstrings = {}

        for func_name in dir(class_obj):
            if (
                callable(getattr(class_obj, func_name))
                and not func_name.startswith("__")
                and not func_name.startswith("update")
                and not func_name.startswith("helper")
                and getattr(class_obj, func_name).__qualname__.startswith(
                    class_obj.__name__ + "."
                )
            ):
                func = getattr(class_obj, func_name)
                if func.__doc__:
                    # Split docstring into lines and process
                    doc_lines = func.__doc__.strip().split("\n")
                    processed_doc = []

                    for line in doc_lines:
                        line = line.strip()
                        if line.startswith("Node Type:") or line.startswith(
                            "Description:"
                        ):
                            processed_doc.append(line)

                    # Join the processed lines back together
                    function_names_and_docstrings[func_name] = "\n".join(processed_doc)
                else:
                    function_names_and_docstrings[func_name] = "No docstring found."

        return function_names_and_docstrings

    def extract_behavior_tree(self, response: str) -> str:
        """
        Extract the behavior tree XML from the response.

        Args:
            response: The response from the LLM

        Returns:
            The extracted behavior tree XML, or an error message
        """
        start_tag = "<BehaviorTree>"
        end_tag = "</BehaviorTree>"
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)

        print(
            f"Current error count: {self.error_count}, Max retries: {self.max_retries}"
        )  # Debug print

        if start_index == -1 or end_index == -1:
            if self.error_count >= self.max_retries:
                print(
                    f"Hit max retries ({self.max_retries}), returning error message"
                )  # Debug print
                return "Could not generate proper XML."
            self.error_count += 1
            print(f"Incrementing error count to {self.error_count}")  # Debug print
            return "try again"

        behavior_tree_xml = response[start_index : end_index + len(end_tag)].strip()
        return behavior_tree_xml

    def _suggest_similar_models(self, model_name: str) -> None:
        """Suggest similar or popular models if the requested model fails to download."""
        popular_models = [
            "llama3.2:latest",
            "llama3.1:latest",
            "llama3:latest",
            "gemma2:latest",
            "mistral:latest",
            "qwen2.5:latest",
        ]

        print("\nSuggested alternative models:")
        for model in popular_models:
            print(f"  - {model}")
        print("\nYou can also check available models at: https://ollama.com/library")
        print("Or list all available models with: ollama list")
