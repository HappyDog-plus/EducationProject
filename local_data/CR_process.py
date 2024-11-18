import pandas as pd
from openai import OpenAI
import re
import ast
import json

def qa_parser(qa_str: str):
    client = OpenAI()
    completion = client.chat.completions.create(
                                                model="gpt-4o-mini",
                                                messages=[
                                                            {
                                                                "role": "system", 
                                                                "content": "Please process the following string and convert it into a list of dictionaries, where each dictionary contains a 'question' and an 'answer' key. Remove any code category identifiers and triple quotes."
                                                            },
                                                            {
                                                                "role": "user",
                                                                "content": qa_str
                                                            }
                                                         ]
                                                )
    output_raw = completion.choices[0].message.content
    return extract_list(output_raw)


# if GPT response content include '[]', this will not work.

# def extract_list(list_str: str):
#     match = re.search(r'(\[.*\])', list_str, re.DOTALL)
#     if match:
#         output_new = match.group(1)
#         parse_res = ast.literal_eval(output_new)
#     else:
#         parse_res = []
#     return parse_res


def extract_list(list_str: str):
    start_index = list_str.find('[')
    end_index = list_str.rfind(']')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return ast.literal_eval(list_str[start_index:end_index + 1])  
    else:
        return []


def img_parse(img_idxs: str):
    img_idxs = img_idxs.replace(" ", "")
    if len(img_idxs) == 0 or img_idxs=="nan":
        return []
    elif '、' in img_idxs:
        res = img_idxs.split('、')
        return  [int(e) for e in res]
    else:
        return [int(img_idxs)]


def main():
    file_path = r"CaseReports.xlsx"
    json_out_path = r"CaseReports.json"
    cat_out_path = r"CaseReportCategories.json"
    # collector = []
    # keys = ["idx", "label", "context", "qa", "imgs"]
    # data = pd.read_excel(file_path).iloc[:, :5]
    # for r in range(len(data)):
    #     d = data.iloc[r]
    #     item = dict.fromkeys(keys, None)
    #     item["idx"] = r
    #     item["label"] = d["Classification"]
    #     item["context"] = d["patient"]
    #     item["qa"] = qa_parser(d["teacher"])
    #     if len(item["qa"]) == 0:
    #         print("idx: ", r, "qa extracting failed.")
    #     item["imgs"] = img_parse(str(d["No. image"]))
    #     collector.append(item)
    #     print("qa ", r, " finished.")
    #     print('-' * 50, "qa", '-'*50)
    #     print(item["qa"])
    #     print('-' * 50, "img ids", '-'*50)
    #     print(item["imgs"])
    #     print('-' * 110, '\n')
    # with open(json_out_path, 'w') as json_file0:
    #     json.dump(collector, json_file0, indent=4)
    # cat_set = set(item["label"] for item in collector)
    # with open(cat_out_path, 'w') as json_file1:
    #     json.dump(list(cat_set), json_file1, indent=4)

    # convert json to excel
    with open(json_out_path, 'r') as f:
        cr_json = json.load(f)
    print("case report amount: ", len(cr_json))
    with open(cat_out_path, 'r') as f:
        cat_json = json.load(f)
    print("case report categories amount:", len(cat_json))


    # start process
    # step 1: truncate questions and answer
    for item in cr_json:
        if len(item["qa"]) > 6:
            item["qa"] = item["qa"][:6]

    # step 2: add qa amount attribute
    n_qa_max = 0
    for item in cr_json:
        item["n_qa"] = len(item["qa"])
        if item["n_qa"] > n_qa_max:
            n_qa_max = item["n_qa"]
    print("n_qa sample: ", cr_json[0])
    print("n_qa max: ", n_qa_max)

    # step 3: split qa attribute and clean string
    def clean_string(text):
        return re.sub(r'^\s*\d+\.\s*', '', text)

    cr_list = []
    for item in cr_json:
        item_new = {
                    "idx": item["idx"],
                    "label": item["label"],
                    "context": item["context"],
                    "imgs": ','.join(map(str, item["imgs"])),
                    "n_qa": item["n_qa"]
                   }
        for i in range(item_new["n_qa"]):
            item_new["q_"+str(i)], item_new["a_"+str(i)] = clean_string(item["qa"][i]["question"]), item["qa"][i]["answer"]
        for j in range(item_new["n_qa"], 6):
            item_new["q_" +str(j)], item_new["a_"+str(j)] = "", ""
        cr_list.append(item_new)

    # step 4: convert to pandas dataframe
    cr_df = pd.DataFrame(cr_list)
    cr_df.to_excel("CR.xlsx", index=False)


if __name__ == "__main__":
    main()
    # input = "Question:  1. What would you tell her about surgical correction of her myopia? 2. What are the complications of LASIK (laser-assisted in situ keratomileusis)? Additional information: this patient undergoes LASIK. The day after surgery she is very happy with her 20/20 vision. On exam her left eye has the finding seen in the photo. 293. What is the diagnosis and what would you do?  Answer: 1. There are a number of surgical options for correcting low myopia. The most common is excisional (laser vision correction [surface ablation, LASIK]). Other techniques include incisional (radial keratotomy) and additive (implants [Intacs]). The indications, risks, benefits, alternatives, and complications of surgery should be discussed as well as the advantages and disadvantages of each of the procedures. She should also be told about what to expect during the preoperative and postoperative periods. 2. LASIK complications include: over- or undercorrection, glare / halos at night, dry eye, irregular / poor flap (too thick or thin, button-hole, incomplete, free cap), epithelial defect, 30decentered ablation, irregular astigmatism, flap dislocation, striae, epithelial ingrowth, interface inflammation (diffuse lamellar keratitis [DLK]), central toxic keratopathy, infection, scarring, and keratectasia. Late DLK may occur (any time in the future) after a corneal abrasion. 3. Stage 2 DLK, which requires frequent topical steroids. Steroid eye drops should be prescribed initially every hour while awake and steroid ointment at bedtime. The eye should be checked daily for improvement, and the steroids are tapered as the DLK resolves. If the interface inflammation progresses to stage 3 or 4, then the flap should be lifted and the stromal bed irrigated. A short course of oral steroids may also be given. "
    # output = qa_parser(input)
    # print(output)



