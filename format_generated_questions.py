import sys
import json
import numpy as np

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Input file:', str(sys.argv[1]))

with open(sys.argv[1]) as f:
    data = json.load(f)


def create_json(questions_json):

    data_json = {"data": []}

    for k, element in questions_json.items():
        title = ""
        context = k.replace("[CTX]", "").replace("[QS]", "")
        qas = []
        for q in element:
            qas.append(q.replace("[QE]", ""))
        paragraphs = []
        questions = []
        for question in qas:
            questions.append({'question': question,
                              'id': str(np.random.randint(100000))})
        paragraphs.append({'context': context,
                           'qas': questions})
        data_json["data"].append({"title": title,
                                  "paragraphs": paragraphs})

    return data_json


formatted_json = create_json(data)

with open("question_dataset.json", "w") as f:
    json.dump(formatted_json, f)
