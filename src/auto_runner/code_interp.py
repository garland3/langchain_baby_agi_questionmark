
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import subprocess
import traceback
# from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

USESUBPROCESS = False

class CodeInterp:
    def __init__(self, run_code=True, print_code=False):
        self.run_code = run_code
        self.print_code = print_code
        self.stdout = ""
        self.stder = ""
        self.my_ipython  = InteractiveShell()

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
        if USESUBPROCESS:
            return self.execute_python_code_subprocess(code, global_vars, local_vars, combine_out)
        else:
            return self.execute_python_code_ipython(code, global_vars, local_vars, combine_out)    
            
    def execute_python_code_ipython(self, code, global_vars={}, local_vars={}, combine_out=True):
        
        
        # capture stdout and stderr
        with io.StringIO() as stdout, io.StringIO() as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # run the cell
                self.my_ipython.run_cell(code)
            # print the captured outputs
            # print("Captured stdout:", stdout.getvalue())
            self.stdout = stdout.getvalue()
            # print("Captured stderr:", stderr.getvalue())
            self.stder = stdout.getvalue()
        
        
    def execute_python_code_subprocess(self, code, global_vars={}, local_vars={}, combine_out=True):
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