import json
import argparse
import openai
from pathlib import Path
import os  # Import os to create directories

# Predefined mapping of objects to numeric IDs
OBJECT_MAP = {
    "bookshelf": 0,
    "chair": 1,
    "desk": 2,
    "door": 3,
    "person": 4,
    # Add more objects here as needed
}

def LM(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0):
    """
    Calls the OpenAI API to generate a response for a given prompt.
    """
    if "gpt" not in gpt_version:
        response = openai.Completion.create(model=gpt_version, 
                                            prompt=prompt, 
                                            max_tokens=max_tokens, 
                                            temperature=temperature, 
                                            stop=stop, 
                                            logprobs=logprobs, 
                                            frequency_penalty=frequency_penalty)
        return response, response["choices"][0]["text"].strip()
    else:
        response = openai.ChatCompletion.create(model=gpt_version, 
                                                messages=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature, 
                                                frequency_penalty=frequency_penalty)
        return response, response["choices"][0]["message"]["content"].strip()

def set_api_key(openai_api_key):
    """
    Sets the OpenAI API key.
    """
    api_key = Path(openai_api_key + '.txt').read_text().strip()
    openai.api_key = api_key

def extract_objects_from_task_llm(task, gpt_version):
    """
    Extracts object names from the task description using the OpenAI model.
    """
    prompt = f"Task: {task}\n\nExtract the names of the objects mentioned in this task. Respond only with the object names separated by commas."
    
    if "gpt" not in gpt_version:
        _, text = LM(prompt, gpt_version, max_tokens=100, stop=["\n"], frequency_penalty=0.15)
    else:
        messages = [{"role": "user", "content": prompt}]
        _, text = LM(messages, gpt_version, max_tokens=100, frequency_penalty=0.15)

    # Parse the response and extract object names
    extracted_objects = [obj.strip().lower() for obj in text.strip().split(",") if obj.strip()]
    return extracted_objects

def map_objects_to_ids(extracted_objects):
    """
    Maps the extracted object names to their corresponding numeric IDs.
    """
    object_ids = {}
    for obj in extracted_objects:
        if obj in OBJECT_MAP:
            object_ids[OBJECT_MAP[obj]] = obj
    return object_ids

def process_tasks_and_generate_objects(test_tasks, gpt_version):
    """
    Process each task and generate the list of objects using LLM.
    """
    task_objects = {}
    for idx, task in enumerate(test_tasks):
        print(f"Processing task {idx+1}: {task}")
        
        # Extract objects mentioned in the task using LLM
        objects_in_task = extract_objects_from_task_llm(task, gpt_version)
        
        # Map object names to their corresponding numeric IDs
        object_ids = map_objects_to_ids(objects_in_task)
        
        # Add the object IDs to the task objects dictionary
        task_objects.update(object_ids)
    
    return task_objects

def get_user_input_tasks():
    """
    Prompts the user to input tasks one by one.
    """
    tasks = []
    print("Please enter your tasks (enter 'done' when finished):")
    while True:
        task = input("Task: ")
        if task.lower() == 'done':
            break
        tasks.append(task)
    return tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gpt-version", type=str, default="gpt-3.5-turbo", 
                        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k'])
    
    args = parser.parse_args()

    # Set the API key
    set_api_key(args.openai_api_key_file)

    # Get user input for tasks
    test_tasks = get_user_input_tasks()

    print(f"----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
    
    # Process tasks and generate object list using LLM
    task_objects = process_tasks_and_generate_objects(test_tasks, args.gpt_version)

    # Ensure the logs directory exists
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Save to JSON
    output_filename = "../logs/PromtObject.json"
    with open(output_filename, "w") as outfile:
        json.dump(task_objects, outfile, indent=4)
    
    print(f"Object list saved to {output_filename}")
