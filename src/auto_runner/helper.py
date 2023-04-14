import os
import subprocess
import sys
import io
import re
import traceback
# import subprocess


def banner(text):
    padding = (30 - len(text)) // 2
    middleline = f"# {' ' * padding}{text}{' ' * padding} #"
    n = len(middleline)
    line = "# " * n

    print(f"{line}\n# {' ' * padding}{text}{' ' * padding} #\n{line}")


def printify(r: str):
    for line in r.split("\n"):
        # if the line is more than 100 chars, then split it into multiple lines.
        n = len(line)
        counter = 0
        while n > 0:
            end_l = min(n, 100)
            currentline = line[0:end_l]
            # tab the line if it is not the first line.
            currentlinestr = currentline if counter == 0 else "\t" + currentline
            print(currentlinestr)
            line = line[end_l:]
            n = len(line)
            counter += 1


class OutputCapture:
    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()

    def __enter__(self):
        sys.stdout = self.out
        sys.stderr = self.err
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if exc_type is not None:
            text = f"An exception occurred: {exc_val}"
            text += "".join(traceback.format_tb(exc_tb))
            print(text)


class CodeSaver:
    def __init__(self):
        pass

    def save_code_dict(self, code_dict: dict):
        # save a dictionary of code to files
        for lang, code in code_dict.items():
            self.save_code(code, lang)

    def save_code(self, code, lang="python",  filename=None):
        if lang == "python":
            suffix = ".py"
        elif lang == "bash":
            suffix = ".sh"
        else:
            raise ValueError(f"lang: {lang} not supported")

       # Save code to a file
        if not os.path.exists('code'):
            os.mkdir('code')
        if filename is None:
            existing_files = [f for f in os.listdir(
                'code') if os.path.isfile(os.path.join('code', f))]
            filename = f"code_{len(existing_files)+1}{suffix}"
        filepath = os.path.join('code', filename)
        with open(filepath, 'w') as f:
            f.write(code)
        return filepath

    def save_output(self, r):
        # save output to file.
        folder = "llm_output"
        if not os.path.exists(folder):
            os.mkdir(folder)
        existing_files = [f for f in os.listdir(
            folder) if os.path.isfile(os.path.join(folder, f))]
        filename = f"llm_output_{len(existing_files)+1}.txt"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w') as f:
            f.write(r)


class LLMTextToCode():
    def __init__(self, run_code=True, print_code=False):
        self.run_code = run_code
        self.print_code = print_code
        self.stdout = ""
        self.stder = ""

    def determine_language(self, codeblock, default_lang="python"):
        lines = codeblock.split("\n")
        line = lines[0].strip()
        # strip the ``` from the line
        line = line.strip('`')
        line = line.strip()
        # strip off the \n
        line = line.strip('\n')
        # if there is nothing left, the assume it is python
        if line == "":
            return default_lang
        lang = line.lower()
        return lang

    def cleanify(self, codeblock):
        # remove the 1st line
        lines = codeblock.split("\n")
        lines = lines[1:]  # strip out the 1st line
        # strip out leading and trailing whitespace
        lines = [l.rstrip() for l in lines]
        lines = [l for l in lines if l != ""]  # remove empty lines
        # strip out leading and trailing backticks
        lines = [l.strip('`') for l in lines]
        if len(lines) == 0:
            return None
        code = "\n".join(lines)
        return code

    def parse_llm_text(self, mytxt):
        # regular expression to find code blocks in markdown
        code_block_regex = re.compile(r'```(.*?)```', re.DOTALL)
        markdown_text = mytxt
        # find all code blocks in markdown
        code_blocks = code_block_regex.findall(markdown_text)
        if len(code_blocks) == 0:
            print("No code, returning")
        # iterate over each code block
        self.lang_code_dict = {}
        for code_block in code_blocks:
            # determine the language of the code block
            lang = self.determine_language(code_block)
            code = self.cleanify(code_block)
            if code is None:
                continue
            # if the language is in the dictionary, then append the code to the existing code
            if lang in self.lang_code_dict:
                self.lang_code_dict[lang] += "\n"+code
            else:
                self.lang_code_dict[lang] = code
        return self.lang_code_dict


