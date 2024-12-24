import pandas as pd
import json
from langchain_openai import ChatOpenAI
import re
import os
import ast

os.environ['OPENAI_API_KEY'] = ""
model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2
                  )

def extract_category(file):
    df = pd.read_excel(file)
    lst = df.iloc[:, 2].dropna().tolist()
    print(len(lst))
    lst = list(set(lst))
    print(len(lst))
    df = pd.DataFrame(lst, columns=['categories'])
    df.to_excel('ExerciseCategories2.xlsx', index=False)

def xlsx_to_json(file):
    df = pd.read_excel(file)
    ex_list = []
    cat_list = []
    course_list = []
    for _, row in df.iterrows():
        categories = row["description"]
        categories = [cat.strip() for cat in categories.split("and")]
        if not len(categories):
            print(row["id"])
        cat_list.extend(categories)
        course_id = ''
        if not pd.isna(row["course_id"]):
            course_id = str(int(row["course_id"]))
            course_list.append(course_id)
        ex_list.append({"id": row["id"], "course_id": course_id, "cats": categories})
    print('*'*20, "Total Exercises", '*'*20, '\n', len(ex_list))
    print('*'*20, "Categories List", '*'*20, '\n', len(cat_list))
    cat_set = list(set(cat_list))
    print('*'*20, "Total Categories", '*'*20, '\n', len(cat_set))
    course_set = list(set(course_list))
    print('*'*20, "Total Course", '*'*20, '\n', course_set)
    with open("categories.json", 'w') as f:
        json.dump(cat_set, f, indent=4)
    with open("exercises.json", 'w') as f1:
        json.dump(ex_list, f1, indent=4)

def match_kwds(input_text):
    input_text = translate_to_english(input_text)
    print(input_text)
    query = input_text
    with open(".\categories.json", 'r') as f:
        kwds = json.load(f)
    prompt = f'''
                You are an expert in ophthalmology, and you need to complete the given tasks strictly in accordance with the format requirements.
                Based on the given list of categories, analyze the following user input, which represents a request for practice questions on a specific concept related to ophthalmology. Your task is to:
                1. Accurately understand the semantic meaning of the input text.
                2. Identify the key ophthalmology-related concepts or keywords from the user's input.
                3. Match these identified keywords to the **most relevant categories** from the provided categories list.
                4. Return at most the top 10 matched categories (maximum), based on their relevance to the user's input. If no matches are found, return an empty list.

                Ensure the returned result strictly adheres to the JSON format as a List.

                Categories list: {', '.join(kwds)}

                User Input Text: {query}

                Return format: [matched categories]
              '''
    response = model.invoke(prompt).content
    print(response)
    matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
    if len(matches):
        response = matches[0]
        matched_kwds = json.loads(response)
        return matched_kwds
    else:
        return []
    
def extract_kwds(text):
    extract_medical_kwds = '''
                                You are a ophthalmology expert. Extract all ophthalmology related keywords from the given text and return them in a format that can be directly parsed as a list.
                                Instructions:
                                1. Only include keywords.
                                2. Return the keywords as a list.
                                3. Do not include any other characters, explanations, or sentences in the output.
                                4. Do not include any words about exercise or practice. It's the context. Not related to ophthalmology. 
                                Input Text:
                                {}
                                Return Format: ["keyword1", "keyword2", "keyword3", ...]
                            '''
    prompt = extract_medical_kwds.format(text)
    response = model.invoke(prompt).content
    # print(response)
    response = re.sub(r"```json|```", "", response)
    print(response)
    parsed_list = ast.literal_eval(response)
    print(parsed_list)
    # print(type(parsed_list))
    return ", ".join(parsed_list)

def translate_to_english(input_text):
    prompt = f'''
                You are a professional translator. If the input text is not in English, translate it into English. If the input text is already in English, return it as is without any changes. Ensure the output is fluent, accurate, and retains the original meaning.

                Input text: "{input_text}"

                Return format: "translated_text"
              '''
    return model.invoke(prompt).content

def match_ex(kwds):
    with open("exercises.json", 'r') as f:
        ex_list = json.load(f)
    return_list = []
    for kwd in kwds:
        for ex in ex_list:
            if kwd in ex["cats"] and ex["id"] not in return_list:
                return_list.append(ex["id"])
    return return_list

if __name__ == "__main__":
    file = r".\Exercises2.xlsx"
    # extract_category(file)
    # xlsx_to_json(file)
    print('*'*20, "Test Match", '*'*20, '\n')
    print(match_ex(match_kwds("我想要一些关于红眼病的练习")))
    pass