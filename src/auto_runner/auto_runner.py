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



from pathlib import Path
from auto_runner.chat_chains import ExecutionChain, Task, TaskCreationChain, TaskPrioritizationChain
from helper import CodeInterp, CodeSaver, LLMTextToCode, banner
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
# pip install faiss-cpu
import faiss


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
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(verbose=False)
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
        self.embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS(self.embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        return vectorstore
        
    def get_info_on_previous_tasks(self, current_task:Task):
        # run the task creation chain
        summary_of_prevous_tasks = _get_top_tasks(self.vectorstore_of_prevous_results, current_task.description, k=5)
        print("Summary of previous tasks: ", summary_of_prevous_tasks)
        return summary_of_prevous_tasks
        
    def run_creation_task(self, task_number:int, previous_result:str):
        # NOTE, the -1
        previous_task = self.task_list[task_number-1]
        
        summary_of_prevous_tasks = self.get_info_on_previous_tasks(previous_task)
        
        print("Running task creation chain")
        
        inputs:Dict = {"objective": self.objective, 
                       "summary_of_prevous_tasks": summary_of_prevous_tasks,
                       "result": previous_result,
                       "prevoius_task_description":previous_task.description, 
                       "incomplete_tasks": [t.description for t in self.task_list[task_number:] ]}
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
    
    def get_task_and_run(self, task_number:int):
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
    
    def reprioritize(self, task_number:int, prevous_task_descrition:str, previous_result:str, number_of_possible_tasks:int):
        # run the task prioritization chain
        print("Running task prioritization chain")
        future_task_descriptions =  "\n".join( [t.description for t in self.task_list[task_number+1:]])
        # print("future_task_descriptions: ", future_task_descriptions)
        inputs:Dict = {"objective": self.objective, 
                       "results": "results", 
                       "task_descriptions":future_task_descriptions, 
                       "last_run_task_description": prevous_task_descrition,
                       "previous_result": previous_result, 
                       "number_of_possible_tasks": number_of_possible_tasks}
        task_list_str = self.task_prioritization_chain.run(inputs)
        # print("Task prioritization chain output: \n", task_list_str)
        
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
        self.task_list = self.task_list[0:task_number+1] + reorderd_old_tasks
        
        # print the reordered old tasks
        print("Reordered old tasks: ")
        for i, t in enumerate(self.task_list):
            if i == self.task_number:
                print(f"\033[1;32;40m{t.description}\033[0m")
            else:
                print("\t" + t.description)
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
            exec_result = self.get_task_and_run(task_number=self.task_number)
            
            # store in vector store
            result_id = f"result_{self.task_number}"
            self.vectorstore_of_prevous_results.add_texts(
                texts=[exec_result],
                metadatas=[{"task_description": self.task_list[self.task_number].description}],
                ids=[result_id],
            )

            
            # base on the results, add more tasks to the task list
            self.run_creation_task(task_number=self.task_number, previous_result=exec_result)
            
            # reprioritize the task list
            possible_tasks_remaining = self.max_runs - loop_counter
            self.reprioritize(task_number=self.task_number, 
                              prevous_task_descrition=self.task_list[self.task_number-1].description, 
                              previous_result=exec_result, number_of_possible_tasks=possible_tasks_remaining)
            
            # ------------------
            # IMPORTANT. INCREMENT TASK NUMBER
            # ------------------
            self.task_number += 1
            print("----------------------------------")
            print("Finished run ", loop_counter, "max loops:", self.max_runs,  " task number ", self.task_number, "number of tasks ", len(self.task_list), " possible_tasks_remaining ", possible_tasks_remaining)
            loop_counter+=1
        
        
        



# OBJECTIVE = "Write a weather report for SF today, and make a predictive model based on the previous 5 days. Using sklearn if needed. Write the report to a file called weather_report.md. "
objective = "Look at the file data.csv. Make a machine learning model to predict temperature. Note the performance of the model. Write a report about the model to a file called model_report.md. I will render it and then give the report to the CEO, so make it clear and look good. "

loopy = AILoop(objective, max_runs=200)
loopy.run()
# %%
