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


class CodeInterp:
    def __init__(self, run_code=True, print_code=False):
        self.run_code = run_code
        self.print_code = print_code
        self.stdout = ""
        self.stder = ""

    def run_codes(self, lang_code_dict,  global_vars={}, local_vars={}):
        # reset the stdout and stderr
        self.stdout = ""
        self.stder = ""
        output = ""
        if "bash" in lang_code_dict:
            bash_code = lang_code_dict.get("bash", "")
            _output = self.execute_bash_code(bash_code)
            output += _output
        if "python" in lang_code_dict:
            python_code = lang_code_dict.get("python", "")
            _output = self.execute_python_code(
                python_code, global_vars, local_vars)
            if _output is not None:
                output += _output

        return output

    def execute_bash_code(self, code):
        # execute bash code
        bash_code = code
        bash_code = bash_code.replace("\n", " && ")
        bash_code = bash_code.replace("&&  &&", "&&")
        # use subprocess to execute the bash code
        print("Running bash code: ", bash_code)
        process = subprocess.Popen(
            bash_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # get the output
        stdout, stderr = process.communicate()
        # add to the stdout and stderr
        self.stdout += stdout.decode("utf-8")
        self.stder += stderr.decode("utf-8")
        r = stdout.decode("utf-8") + stderr.decode("utf-8")
        print("Finished running bash code")
        return r

    def execute_python_code(self, code, global_vars={}, local_vars={}, combine_out=True):
        self.local_vars = local_vars
        self.global_vars = global_vars
        try:
            # make a temp folder called _temp
            if not os.path.exists('_temp'):
                os.mkdir('_temp')
            # save the code to a file
            filename = "_temp/temp.py"
            with open(filename, 'w') as f:
                f.write(code)
            python_interpreter = sys.executable
            print(f"python_interpreter: {python_interpreter}")

            # run autopep8 on the file 1st.
            p_auto8 = subprocess.Popen([python_interpreter, "-m", "autopep8",
                                       "--in-place", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out1, error1 = p_auto8.communicate()
            self.stdout += out1.decode("utf-8")
            self.stder += error1.decode("utf-8")

            process = subprocess.Popen(
                [python_interpreter, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # get the output
            stdout, stderr = process.communicate()
            self. stdout += stdout.decode("utf-8")
            self.stder += stderr.decode("utf-8")
            return self.stdout + self.stder

            # with OutputCapture() as capture:
            #     exec(code, self.global_vars, self.local_vars)
            #     self.stdout = capture.out.getvalue()
            #     self.stder = capture.err.getvalue()
            #     if combine_out:
            #         return capture.out.getvalue()+" "+ capture.err.getvalue()
            #     return capture.out.getvalue(), capture.err.getvalue()
        except Exception as e:
            print(e)
            self.stdout = ""
            self.stder = str(e) + traceback.format_exc()
        print("Finished running python code")

    def clean_stderr(self):
        return self.stder.strip().strip("\n")

    def clean_stdout(self, max_len=1200, snip_len=300):
        t = self.stdout.strip().strip("\n")
        # count the number of words and white spaces, if greater than 2500, then truncate, by keeping the first and last 500, put dotsa in the middle and say `truncate`
        n = len(t.split())+len(t.split(" "))
        if n > max_len:
            print("Truncating")
            t = " ".join(t.split()[:500]) + \
                " ... TRUNCATED ... " + " ".join(t.split()[-500:])
            return t
        else:
            return t

    def verify_output_no_error(self):
        stderr = self.clean_stderr()
        if stderr != "":
            return False
        return True
