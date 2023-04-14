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
    def __init__(self, chat_session: ChatSession, sys_message: str = "", default_prompt: str = "", input_list=None, verbose=False):
        self.chat_session = chat_session
        self.sys_message = sys_message
        self.default_prompt = default_prompt
        self.input_list = input_list if input_list is not None else []
        self.verbose = verbose

        if self.sys_message:
            self.chat_session.send_message(self.sys_message)

    def run(self, inputs_as_dict: dict) -> str:
        formatted_prompt = self.default_prompt.format(**inputs_as_dict)
        if self.verbose:
            print("Prompt is:\n"+formatted_prompt)
        response = self.chat_session.send_message(formatted_prompt)
        return response




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
            " This result was based on this task description: {prevoius_task_description}."
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
            input_list=["result", "prevoius_task_description", "incomplete_tasks", "objective"],
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
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing\n"
            " the following tasks\n: {task_descriptions}.\n\n"
            " Consider the ultimate objective of your team: {objective}.\n"
            " At max return 10 tasks. This might be less if you think some tasks are not needed.\n"
            " You can return fewer tasks if you think some tasks are not needed."
            "Some combining of tasks is allowed.\n"
            "The most recently run task description was {last_run_task_description}.\n"
            "The output of the most recently run task was \n---------\n{previous_result}.\n---------\n\n"
            "We have limited time. AT max,there are only {number_of_possible_tasks} more tasks that we have time to complete.\n"
            "You need to help future tasks understand what the previous tasks were.\n"
            "In particular be specific about which files to save and which files to load.\n"
            "You must always save results from one task to disk so that the next task can load it.\n"
            
            "  Return the result as a bullet list, like:"
            " * First task"
            " * Second task"
            # " Start the task list with number {next_task_id}."
        )
        chat_session = ChatSession()
        chat_session_wrapper = cls(
            chat_session=chat_session,
            default_prompt=task_prioritization_template,
            input_list=["task_descriptions",  "objective", "last_run_task_description", "previous_result", "number_of_possible_tasks"],
            verbose=verbose
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
            "I would also recommend that you save scratch results to mylog.txt. Write to mylog.txt as needed. And read it as needed. "   
            
            "Also, it is essential that you save intermediate results to disk."
            " Always use a print statement to indicate the name of the files you save and what is in the file."         
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
            stderror = ci.clean_stderr()
            if stderror != "":
                print("Error found!")
                print(f"\033[1;31;40m{stderror}\033[0m")
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
        text += "\nAfter running the code, The std error is:\n"+ ci.clean_stderr() + "\nThe stdout is:\n" + ci.clean_stdout()
        return text
        

class Task(BaseModel):
    name: str = Field(..., description="The name of the task")
    description: str = Field(..., description="The description of the task")
