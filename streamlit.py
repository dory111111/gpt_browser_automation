import re

import streamlit as st
from langchain import PromptTemplate, OpenAI
from langchain.chains import PALChain
from selenium import webdriver
from selenium.webdriver.common.by import By
from contextlib import contextmanager, redirect_stdout

from io import StringIO

@contextmanager
def st_capture(output_function):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_function(escape_ansi(stdout.getvalue()))
            return ret
        
        stdout.write = new_write
        yield

def escape_ansi(line):
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


template = """If someone asks you to perform a task, your job is to come up with a series of Selenium Python commands that will perform the task. 
There is no need to need to include the descriptive text about the program in your answer, only the commands.
Note that the version of selenium is 4.7.2.
find_element_by_class_name is deprecated.
Please use find_element(by=By.CLASS_NAME, value=name) instead.
You must use detach option when webdriver
You must starting webdriver with --lang=en-US

Begin!
Your job: {question}
"""

st.set_page_config(layout="wide") 
st.title('🐧 Demo: Using GPT-3 for Browser Automation')
col1, col2 = st.columns(2)

with col1:
    openai_api_key = st.text_input(label="OpenAI API key", placeholder="Input your OpenAI API key here:",type="password")
    question = st.text_area(
        label = "Input" ,
        placeholder = "e.g. Go to https://www.google.com/ and search for GPT-3"
    )
    start_button = st.button('Run')
            
with col2:
    if start_button:
        with st.spinner("Running..."):
            llm=OpenAI(temperature=0,openai_api_key=openai_api_key)
            chain = PALChain.from_colored_object_prompt(llm, verbose=True)
            output = st.empty()
            with st_capture(output.code):
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["question"]
                )
                prompt = prompt.format(question=question)
                chain.run(prompt)