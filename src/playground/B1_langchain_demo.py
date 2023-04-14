# %% [markdown]
import os
from collections import deque
from typing import Dict, List, Optional, Any
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from llminterface.llm_wrapper import ChatSession
# get OPENAI_API_KEY from env var, use s
import os
import tomli
from pathlib import Path
from helper import CodeInterp, CodeSaver, LLMTextToCode, banner
import os
import datetime
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
# pip install faiss-cpu
import faiss

# read ~/.llminteface/.secrets.toml
# get the openaikey value. 
with open(str(Path.home() / '.llminterface/.secrets.toml'), 'rb') as f:
    secrets = tomli.load(f)
    openai_key = secrets['openaikey']
    print(openai_key)
    os.environ['OPENAI_API_KEY'] = openai_key

# 
# Depending on what vectorstore you use, this step may look different.


class ChatSessionWrapper:
    def __init__(self, chat_session: ChatSession, sys_message: str = "", default_prompt: str = "", input_list=None):
        self.chat_session = chat_session
        self.sys_message = sys_message
        self.default_prompt = default_prompt
        self.input_list = input_list if input_list is not None else []

        if self.sys_message:
            self.chat_session.send_message(self.sys_message)

    def run(self, inputs_as_dict: dict) -> str:
        formatted_prompt = self.default_prompt.format(**inputs_as_dict)
        response = self.chat_session.send_message(formatted_prompt)
        return response

# # Usage example:
# chat_session = ChatSession(name="my_chat_session")
# wrapper = ChatSessionWrapper(
#     chat_session,
#     sys_message="I am an AI assistant that can answer questions and provide useful information.",
#     default_prompt="What is the capital of {country}?",
#     input_list=["country"]
# )

# response = wrapper.run({"country": "France"})
# print(response)

# %%
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# %% [markdown]
# ## Define the ChatWrappers
# 
# BabyAGI relies on three ChatWrappers:
# - Task creation chain to select new tasks to add to the list
# - Task prioritization chain to re-prioritize tasks
# - Execution Chain to execute the tasks

# %%

# %%
class TaskCreationChain(ChatSessionWrapper):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, verbose = False) -> ChatSessionWrapper:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last task run has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
            "Only return the new tasks."
            "Do not return any commentary on the tasks"
            "Your response MUST only be bullet points."
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=task_creation_template,
            input_list=["result", "task_description", "incomplete_tasks", "objective"],
        )

        return chat_session_wrapper

tc = TaskCreationChain.from_llm()
result = tc.run({"result": "nothing", "task_description": "catch a fish", "incomplete_tasks": "start", "objective": "cath a fish"})
result

# %%
class TaskPrioritizationChain(ChatSessionWrapper):
    """Chain to prioritize tasks."""
    @classmethod
    def from_llm(cls, verbose = False) -> ChatSessionWrapper:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=task_prioritization_template,
            input_list=["task_names", "next_task_id", "objective"],
        )
        return chat_session_wrapper

# %%
class CodeFixerChain(ChatSessionWrapper):

    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, verbose = False) -> ChatSessionWrapper:
        """Get the response parser."""
        code_fixer_template = (
            "```python\n {code}.```\n\n"
            "You are a code fixer AI tasked with fixing the code:\n"
            "The code was run and has this stderr output: {stderr}.\n"
            "You must always return  python code and nothing else.\n"
            "If you needed to call a system function, then you can use the `os` module.\n"
            "Do NOT return markdown, but only return python code.\n"
            "Your code :\n\n"            
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=code_fixer_template,
            input_list=["code", "stderr"],
        )
        return chat_session_wrapper
# %%

class ExecutionChain(ChatSessionWrapper):

    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, verbose = False) -> ChatSessionWrapper:
        """Get the response parser."""
        execution_template = (
            "You are an AI tasked with performing the following objective: {objective}. "
            "Here is some context on previous tasks completed: {context}. "
            "Some of those tasks may be incomplete. If code is involved it may have errors. So please verify a previous step was finished before moving on. "
            "If you include ```python\ncode\n``` in your response, it will be executed. I can only use code if it is in a python code block."
            "`mycode` will not be executed. You must always say it is python code."
            "Feel free to use Python as needed to complete your task. "
            "Your task is: {task}. "
            "Response:"
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=execution_template,
            input_list=["objective", "context", "task"]
        )
        return chat_session_wrapper

    def run(self, *args, **kwargs):
        # call the super class's run method
        print('run', args, kwargs)
        text = super().run(*args, kwargs)

        llmtext2code = LLMTextToCode()
        lang_code_dict = llmtext2code.parse_llm_text(text)
        if len(lang_code_dict) == 0:
            return text
        
        cs = CodeSaver()
        cs.save_code_dict(lang_code_dict)
        
        print("Code found!")
        ci = CodeInterp()
        ci.run_codes(lang_code_dict)
        for j in range(5):
            print("Fixing code iteration: ", j)
            if ci.clean_stderr() != "":
                print("Error found!")
                codefixer = CodeFixerChain.from_llm()
                python_code = codefixer.run(code=lang_code_dict['python'], stderr=ci.clean_stderr())
                print("Code fixer output: ", python_code, "\n\n")
                fixer_lang_code_dict = {"python": python_code}
                # fixer_lang_code_dict = llmtext2code.parse_llm_text(output_code)
                cs.save_code_dict(fixer_lang_code_dict)
                ci.run_codes(fixer_lang_code_dict)
            else:
                banner("Code works!")
                break
        text += "\nAfter running the code std error:\n\n"+ ci.clean_stderr() + "\n\n std out:\n\n" + ci.clean_stdout()
        return text
        


# %% [markdown]
# ### Define the BabyAGI Controller
# 
# BabyAGI composes the chains defined above in a (potentially-)infinite loop.

# %%
def get_next_task(task_creation_chain: TaskCreationChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

# %%
def prioritize_tasks(task_prioritization_chain: ChatSessionWrapper, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list

# %%
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task']) for item in sorted_results]

def execute_task(vectorstore, execution_chain: LLMChain, objective: Dict, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective['objective'], k=k)
    return execution_chain.run(objective=objective, context=context, task=task)

# %%

class BabyAGI(BaseModel ):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
        
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)
        
    @property
    def input_keys(self) -> List[str]:
        return ["objective"]
    
    @property
    def output_keys(self) -> List[str]:
        return []

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = {'objective':inputs['objective']}
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls,     
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm()
        task_prioritization_chain = TaskPrioritizationChain.from_llm()
        execution_chain = ExecutionChain.from_llm()
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs
        )

# %% [markdown]
# ### Run the BabyAGI
# 
# Now it's time to create the BabyAGI controller and watch it try to accomplish your objective.

# %%
# OBJECTIVE = "Write a weather report for SF today, and make a predictive model based on the previous 5 days. Using sklearn if needed. Write the report to a file called weather_report.md. "
OBJECTIVE = "Look at the file data.csv. Make a machine learning model to predict temperature. Note the performance of the model. Write a report about the model to a file called model_report.md. I will render it and then give the report to the CEO, so make it clear and look good. "

# %%
# llm = OpenAI(temperature=0, max_tokens= 1000)

# %%
# Logging of LLMChains
verbose=False
# If None, will keep on going forever
max_iterations: Optional[int] = 30
baby_agi = BabyAGI.from_llm(  
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)

# %%
baby_agi({"objective": OBJECTIVE})

# %%



