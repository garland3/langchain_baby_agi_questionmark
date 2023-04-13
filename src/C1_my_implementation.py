# %% [markdown]
import os
from typing import Dict, List, Optional, Any
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from llminterface.llm_wrapper import ChatSession
# get OPENAI_API_KEY from env var, use s
import os
import tomli
from pathlib import Path
from helper import CodeInterp, CodeSaver, LLMTextToCode, banner
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
# pip install faiss-cpu
import faiss

# read ~/.llminteface/.secrets.toml
# get the openaikey value. 
with open(str(Path.home() / '.llminterface/.secrets.toml'), 'rb') as f:
    secrets = tomli.load(f)
    openai_key = secrets['openaikey']
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

# %%
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})



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

# tc = TaskCreationChain.from_llm()
# result = tc.run({"result": "nothing", "task_description": "catch a fish", "incomplete_tasks": "start", "objective": "cath a fish"})
# result

# %%
class TaskPrioritizationChain(ChatSessionWrapper):
    """Chain to prioritize tasks."""
    @classmethod
    def from_llm(cls, verbose = False) -> ChatSessionWrapper:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_descriptions}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a bullet list, like:"
            " * First task"
            " * Second task"
            # " Start the task list with number {next_task_id}."
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=task_prioritization_template,
            input_list=["task_descriptions",  "objective"],
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
            "The code was run and has this stdout output: {stdout}.\n"
            "You can run python or bash code. You must use code blocks to run code. Inline code will not work. \n"
            "As your expert advisor. I would recommend that you add lots of print statements to help you debug your code and help future tasks."            
                        "Your code :\n\n"            
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=code_fixer_template,
            input_list=["code", "stderr", "stdout"],
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
            # "Some of those tasks may be incomplete. If code is involved it may have errors. So please verify a previous step was finished before moving on. "
            "If you include ```python\ncode\n``` in your response, it will be executed. I can only use code if it is in a python code block."
            "`mycode` will not be executed. You must always say it is python code in a code block."
            "Feel free to use Python as needed to complete your task."
            "As your expert advisor. I would recommend that you add lots of print statements to help you debug your code and help future tasks."            
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
        # print('run', args, kwargs)
        text = super().run(*args,**kwargs)

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
                
                code_as_str ="\n\n".join( [f"{lang}\n\n{code}\n\n" for lang,code  in lang_code_dict.items() ])
                inputs_dict = {"code": code_as_str,  "stderr": ci.clean_stderr(), "stdout": ci.clean_stdout()}
                output_code = codefixer.run(inputs_as_dict=inputs_dict)
                print("Code fixer output: \n\n------------\n", output_code, "\n\n")
                fixer_lang_code_dict = llmtext2code.parse_llm_text(output_code)
                cs.save_code_dict(fixer_lang_code_dict)
                ci.run_codes(fixer_lang_code_dict)
            else:
                banner("Code works!")
                break
        text += "\nAfter running the code std error:\n\n"+ ci.clean_stderr() + "\n\n std out:\n\n" + ci.clean_stdout()
        return text
        

class Task(BaseModel):
    name: str = Field(..., description="The name of the task")
    description: str = Field(..., description="The description of the task")

# %%
task = Task(name="task1", description="description1")
print(task)

# %%

# %%

def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task_description']) for item in sorted_results]


class AILoop:
    def __init__(self, objective, max_runs = 30):
        # make a TaskCreationChain, TaskPrioritizationChain, ExecutionChain
        self.task_creation_chain = TaskCreationChain.from_llm()
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm()
        self.execution_chain = ExecutionChain.from_llm()
        # set the objective
        self.objective = objective
        # make task list as a deque
        self.task_list : List[Task] = []
        self.task_number = 0
        self.max_runs = max_runs
        self.vectorstore_of_prevous_results:VectorStore = self.make_vector_store()
        
    def make_vector_store(self):
        import faiss
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        return vectorstore
        
    def get_info_on_previous_tasks(self, current_task:Task):
        # run the task creation chain
        summary_of_prevous_tasks = _get_top_tasks(self.vectorstore_of_prevous_results, current_task.description, k=5)
        print("Summary of previous tasks: ", summary_of_prevous_tasks)
        return summary_of_prevous_tasks
        
    def run_creation_task(self, task_number:int, previous_result:str):
        task = self.task_list[task_number]
        
        summary_of_prevous_tasks = self.get_info_on_previous_tasks(task)
        
        print("Running task creation chain")
        
        inputs:Dict = {"objective": self.objective, 
                       "summary_of_prevous_tasks": summary_of_prevous_tasks,
                       "result": previous_result,
                       "task_description":task.description, 
                       "incomplete_tasks": [t.name for t in self.task_list[task_number:] ]}
        task_description = self.task_creation_chain.run(inputs)
        print("Task creation chain output: \n", task_description)
        # add the task to the task list
        # split by new line, strip, and remove empty strings, add new task
        lines = [line.strip() for line in task_description.splitlines() if line.strip() != ""]
        for l in lines:
            self.task_list.append(Task(name=f"task{self.task_number}", description=l))
            # self.task_number += 1
        
        # return the task description
        return task_description
    
    def get_next_task_and_run(self, task_number:int):
        # get the task
        task = self.task_list[task_number]
        # run the execution chain
        print("Running execution chain. Task Description: ", task.description)
        inputs:Dict = {"objective": self.objective, 
                       "context": "context", 
                       "task": task.description}
        result = self.execution_chain.run(inputs_as_dict=inputs)
        print("Execution chain output: \n", result)
        # return the result
        return result
    
    def reprioritize(self, task_number:int):
        # run the task prioritization chain
        print("Running task prioritization chain")
        inputs:Dict = {"objective": self.objective, 
                       "results": "results", 
                       "task_descriptions": [t.description for t in self.task_list[task_number:]]}
        task_list_str = self.task_prioritization_chain.run(inputs)
        print("Task prioritization chain output: \n", task_list_str)
        
        # the new self.task_list has old tasks + new tasks
        lines = [line.strip() for line in task_list_str.splitlines() if line.strip() != ""]
        # only keep those lines which have a *, -, or number in front of them
        lines = [line[1:] for line in lines if line[0] in ["*", "-", "1", "2", "3", "4", "5", "6", "7", "8", "9"]]
        # lines = [line[1:] for line in lines if line[0] == "*"]
        reorderd_old_tasks = []
        for l in lines:
            reorderd_old_tasks.append(Task(name=f"task{self.task_number}", description=l))
            # self.task_number += 1
            
        # OLD TASKS + NEW TASKS
        self.task_list = self.task_list[:task_number] + reorderd_old_tasks
        return task_list_str
        
        
    def run(self):
        # add a starting task to the task list
        mytask = Task(name="start", description="start")
        self.task_list.append(mytask)
        
        # inputs are teh objective, results, task_description, incomplete_tasks
        self.run_creation_task(task_number=0,  previous_result="")
        self.task_number = 1 # push it to 1
        loop_counter = 0
        while loop_counter < self.max_runs and self.task_number < len(self.task_list):
            # Get the task, and run it. 
            exec_result = self.get_next_task_and_run(task_number=self.task_number)
            
            # store in vector store
            result_id = f"result_{self.task_number}"
            self.vectorstore_of_prevous_results.add_texts(
                texts=[exec_result],
                metadatas=[{"task_description": self.task_list[self.task_number].description}],
                ids=[result_id],
            )

            self.task_number += 1
            
            # base on the results, add more tasks to the task list
            self.run_creation_task(task_number=self.task_number, previous_result=exec_result)
            
            # reprioritize the task list
            self.reprioritize(task_number=self.task_number)
            
            print("----------------------------------")
            print("Finished run ", loop_counter, " task number ", self.task_number, "number of tasks ", len(self.task_list))
            loop_counter+=1
        
        
        



# OBJECTIVE = "Write a weather report for SF today, and make a predictive model based on the previous 5 days. Using sklearn if needed. Write the report to a file called weather_report.md. "
objective = "Look at the file data.csv. Make a machine learning model to predict temperature. Note the performance of the model. Write a report about the model to a file called model_report.md. I will render it and then give the report to the CEO, so make it clear and look good. "

loopy = AILoop(objective)
loopy.run()