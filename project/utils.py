import json
from deep_translator import GoogleTranslator

def generateHistory():
    context = open_json("project/contexts.json")
    conversations = []
    for item in context:
            for key, value in item.items():
                # Divide cada fala pelo separador "|"
                parts = [x.strip() for x in value.split("|") if x.strip()]
                conversations.append(parts)
    return conversations


def generateLabels():
    history = generateHistory()
    labels = []
    for pos,item in enumerate(history):
        for sub in item:
            labels.append(pos)
    print(labels)
    print(history)
    return labels

def open_json(path):
    contextData = None
    with open(path, 'r', encoding='utf-8') as f:
        contextData = json.load(f)

    return contextData



def translator(source, target, txt_input):
    return GoogleTranslator(source = source, target = target).translate(txt_input)