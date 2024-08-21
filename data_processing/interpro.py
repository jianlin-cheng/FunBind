import xml.etree.ElementTree as ET
import re
import pickle


def save_pickle(obj, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error saving object: {e}")


def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {filepath}.")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None
    

def extract_text(element):
    temp = ''.join(element.itertext())
    temp = temp.strip()
    temp = ' '.join(temp.split())
    temp = re.sub(r'\[.*?\]', '', temp)
    temp = re.sub(r'\s+', ' ', temp)
    temp = re.sub(r'\s+([.,!?;:])', r'\1', temp)
    return temp

def clean_text(text):
    return ' '.join(text.split())


def extract_abstracts(file_path):
    
    abstracts = {}
    tree = ET.parse(file_path)
    root = tree.getroot()

    for entry in root.findall('interpro'):
        abstract = entry.find('abstract')
        if abstract is not None:
            abstract_text = extract_text(abstract)
            cleaned_text = clean_text(abstract_text)

            abstracts[entry.get('id')] = [entry.get('short_name'), entry.get('type'), 
                                          entry.get('is-llm'), entry.get('is-llm'), 
                                          cleaned_text]
    return abstracts


def main():
    file_path = "D:\Workspace\python-3\FunBindWorkspace\FunBindData\interpro.xml"
    # abstracts = extract_abstracts(file_path)

    # save_pickle(abstracts, "D:/Workspace/python-3/FunBindWorkspace/FunBindData/abtsracts.pkl")


    data = load_pickle("D:/Workspace/python-3/FunBindWorkspace/FunBindData/abtsracts.pkl")

    for i in data:
        print(i, data[i])
        exit()



if __name__ == "__main__":
    main()