from datasets import load_dataset
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def get_docstring_node(node):
    docstring_node = []
    # traverse_type(node, docstring_node, kind=['expression_statement']) #, 'comment'])
    for child in node.children:
        if child.type == 'block':
            for sub_child in child.children:
                if sub_child.type == 'expression_statement':
                    docstring_node.append(sub_child)

    docstring_node = [node for node in docstring_node if
                        node.type == 'expression_statement' and node.children[0].type == 'string']
    
    if len(docstring_node) > 0:
        return [docstring_node[0].children[0]]  # only take the first block

    return None

def get_target_function_prompt(code):
    code_encode = bytes(code, "utf8")
    root = parser.parse(code_encode)
    root_node = root.root_node

    docstring_list = get_docstring_node(root_node.children[0])
    end_node = None
    if docstring_list:
        end_node = docstring_list[0]
    else:
        for child_node in root_node.children[0].children:
            if child_node.type == ":":
                end_node = child_node
                break

    assert end_node is not None

    lines = code_encode.splitlines() 
    lines = lines[:(end_node.end_point[0]+1)]
    lines[-1] = lines[-1][:end_node.end_point[1]]
    lines = [x.decode("utf8") for x in lines]
    prompt = "\n".join(lines)
    return prompt