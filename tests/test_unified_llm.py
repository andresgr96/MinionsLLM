"""
Test script for the unified BehaviorTreeGenerator class.
This demonstrates how to use both llamacpp and ollama backends.
"""

import os
import sys
import time
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_layer import BehaviorTreeGenerator
from control_layer.simulation.agents.robot_agent import RobotAgent

def test_llamacpp_backend():
    """Test the llamacpp backend (original functionality)"""
    print("=== Testing llamacpp Backend ===")
    
    try:
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="llamacpp",
            model_path_or_url="./llm_layer/models/Llama-3.1-8B-Instruct-BT-B2-1000.Q8_0.gguf",
            chat_format="llama-3",
            context_length=1024*4,
            gpu_layers=0,
            temperature=0.2,
            verbose=False
        )
        
        prompt = "Find all the parts in the environment. If you find a good part, go to the base. If you are in the base then drop it there. If you find a scrap part, go to the waste. If you are in the waste then drop it there."
        
        start_time = time.time()
        error_count, behavior_tree = generator.generate_behavior_tree(
            prompt=prompt,
            which_prompt=3,  # Two-shot
            log_prompt=False
        )
        end_time = time.time()
        prompting_time = end_time - start_time
        
        print(f"LlamaCPP Result:")
        print(f"Error count: {error_count}")
        print(f"Prompting time: {prompting_time:.2f} seconds")
        print(f"Generated tree:\n{behavior_tree}")
        
    except Exception as e:
        print(f"LlamaCPP Backend Error: {e}")

def test_ollama_backend_existing_model():
    """Test the ollama backend with an existing model"""
    print("\n=== Testing Ollama Backend (Existing Model) ===")
    
    try:
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama",
            ollama_model_name="gemma3:1b",  # Use proper Ollama model format
            temperature=0.0
        )
        
        prompt = "Find all the parts in the environment. If you find a good part, go to the base. If you are in the base then drop it there. If you find a scrap part, go to the waste. If you are in the waste then drop it there."
        
        start_time = time.time()
        error_count, behavior_tree = generator.generate_behavior_tree(
            prompt=prompt,
            which_prompt=3,  # Two-shot
            log_prompt=True
        )
        end_time = time.time()
        prompting_time = end_time - start_time
        
        print(f"Ollama (Existing Model) Result:")
        print(f"Error count: {error_count}")
        print(f"Prompting time: {prompting_time:.2f} seconds")
        print(f"Generated tree:\n{behavior_tree}")
        
    except Exception as e:
        print(f"Ollama (Existing Model) Backend Error: {e}")

def test_ollama_backend_auto_import():
    """Test the ollama backend with auto-import of GGUF file"""
    print("\n=== Testing Ollama Backend (Auto-Import GGUF) ===")
    
    try:
        generator = BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama",
            model_path_or_url="hf.co/Andresgr96/gemma-3-1b-it-qat-A:Q8_0",  # Auto-import this model
            # ollama_model_name="hf.co/Andresgr96/Llama-3.1-8B-Instruct-BT-B2-1000:Q8_0",
            temperature=0.2
        )
        
        prompt = "Find all the parts in the environment. If you find a good part, go to the base. If you are in the base then drop it there. If you find a scrap part, go to the waste. If you are in the waste then drop it there."
        
        start_time = time.time()
        error_count, behavior_tree = generator.generate_behavior_tree(
            prompt=prompt,
            which_prompt=3,  # Two-shot
            log_prompt=False
        )
        end_time = time.time()
        prompting_time = end_time - start_time
        
        print(f"Ollama (Auto-Import) Result:")
        print(f"Error count: {error_count}")
        print(f"Prompting time: {prompting_time:.2f} seconds")
        print(f"Generated tree:\n{behavior_tree}")
        
    except Exception as e:
        print(f"Ollama (Auto-Import) Backend Error: {e}")

def test_parameter_validation():
    """Test parameter validation"""
    print("\n=== Testing Parameter Validation ===")
    
    # Test missing model for llamacpp
    try:
        BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="llamacpp"
            # Missing model_name_or_path
        )
    except ValueError as e:
        print(f"✓ Caught expected error for llamacpp: {e}")
    
    # Test missing model for ollama
    try:
        BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="ollama"
            # Missing both model_name_or_path and ollama_model_name
        )
    except ValueError as e:
        print(f"✓ Caught expected error for ollama: {e}")
    
    # Test invalid backend
    try:
        BehaviorTreeGenerator(
            agent_class=RobotAgent,
            backend="invalid_backend",
            model_name_or_path="some_model"
        )
    except ValueError as e:
        print(f"✓ Caught expected error for invalid backend: {e}")

def main():
    """Run all tests"""
    print("Testing Unified BehaviorTreeGenerator Class")
    print("=" * 50)
    
    # Test parameter validation first
    # test_parameter_validation()
    
    # test_ollama_backend_existing_model()
    test_ollama_backend_auto_import()
    # test_llamacpp_backend()

    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo test the backends:")
    print("1. For LlamaCPP: Ensure llama-cpp-python is installed and uncomment test_llamacpp_backend()")
    print("2. For Ollama: Ensure Ollama is installed and running, then uncomment the ollama tests")

if __name__ == "__main__":
    main() 