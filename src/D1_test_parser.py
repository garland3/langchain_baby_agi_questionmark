# %%
from helper import CodeInterp, CodeSaver, LLMTextToCode, banner

file = r"/home/garlan/git/llms/langchainbabyagi/code/code_12.py"

txt = open(file).read()

txt2code = LLMTextToCode()
r = txt2code.parse_llm_text(txt)
# %%
r
# %%
print(r['python'])
# %%
