import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import networkx as nx
import re



def remove_newlines_without_semicolons(text):
    # Define a regular expression pattern to match lines without semicolons
    pattern = r'([^;])\n'
    # Replace newline characters with a space for lines not ending with semicolons
    result = re.sub(pattern, r'\1 ', text)
    return result

def remove_semicolons(text):
    lines = text.split('\n')  # Split the text into lines
    cleaned_lines = [line.rstrip(';') for line in lines]  # Remove semicolons from each line
    cleaned_text = '\n'.join(cleaned_lines)  # Join the lines back together
    return cleaned_text

def remove_multiple_spaces(text):
    # Use regular expression to replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text


def separate_variable_names_with_sizes(text):
    # Split the string into individual variable names
    #variable_names = [var.split('[')[0] for var in text.split(',')]
    variable_names = [var.split('[')[0].strip() for var in text.split(',')]
    # Count the occurrences of each variable name
    variable_counts = {}
    for name in variable_names:
        variable_counts[name.replace(" ", "")] = variable_counts.get(name.replace(" ", ""), 0) + 1
    
    return variable_counts


def parse_verilog_module(verilog_code):
    refined_code = remove_newlines_without_semicolons(verilog_code)
    refined_code = remove_semicolons(refined_code)
    lines = refined_code.split('\n')
    inputs = []
    outputs = []
    
    for line in lines:
        
        if 'input' in line:
            #inputs = remove_multiple_spaces("".join(line.split(","))).split('\\')[1:]
            inputs = separate_variable_names_with_sizes(" , ".join(remove_multiple_spaces("".join(line.split(","))).split('\\')[1:]))
        elif 'output' in line:
            #outputs = remove_multiple_spaces("".join(line.split(","))).split('\\')[1:]
            outputs = separate_variable_names_with_sizes(" , ".join(remove_multiple_spaces("".join(line.split(","))).replace("\\", "").split()[1:]))
    num_inputs = list(inputs.values())
    num_outputs = list(outputs.values())
    return list(inputs.keys()), list(outputs.keys()), num_inputs, num_outputs

def generate_verilog_pytorch_model(module_name, inputs, outputs, num_outputs, verilog_code):
    assignments = re.findall(r'assign\s+(\S+)\s*=\s*(\S+)\s*([&|^])\s*(\S+)\s*;|assign\s+(\S+)\s*=\s*(\S+)\s*;', verilog_code.replace("\\", "") )
    class_name = module_name.lower()
    class_definition = f"class {class_name}(nn.Module):\n" \
                       f"    def __init__(self):\n" \
                       f"        super().__init__()\n\n" \
                       f"    def forward(self, {', '.join(inputs)}):\n"
    '''for i in range(len(outputs)):
        class_definition += f"        {outputs[i]} = torch.zeros({inputs[0]}.shape[0], {num_outputs[i]}).to({inputs[0]}.device)\n"'''
    #f"        {', '.join(outputs)} = torch.zeros({inputs[0]}.shape[0], {len(outputs)}).to({inputs[0]}.device)\n\n" \
    # cnt = 0
    for assignment in assignments:
        
        var_name, operand1, operator, operand2, var, operand = assignment
        if var == '':
            if re.match(r"~?(\w+)", operand1).group(1) in inputs:
                operand1 = operand1.replace("[", "[:,")
                operand1 = operand1.replace("]", "].unsqueeze(-1)")
            if re.match(r"~?(\w+)", operand2).group(1) in inputs:
                operand2 = operand2.replace("[", "[:,")
                operand2 = operand2.replace("]", "].unsqueeze(-1)")
            if re.match(r"~?(\w+)", operand1).group(1) in outputs:
                operand1 = operand1.replace('[', '_').replace(']', '')
            if re.match(r"~?(\w+)", operand2).group(1) in outputs:
                operand2 = operand2.replace('[', '_').replace(']', '')
            if re.match(r"~?(\w+)", var_name).group(1) in outputs:
                var_name = var_name.replace('[', '_').replace(']', '')
            # if cnt == 10:
            #     sdasdasd
            # cnt += 1
            pytorch_script = f"        {var_name} = AND ( {'NOT( ' + operand1[1:] + ' )' if operand1.startswith('~') else operand1} , {'NOT( ' + operand2[1:] + ' )' if operand2.startswith('~') else operand2} )"
            if operator == '|':
                pytorch_script = f"        {var_name} = OR ( {'NOT( ' + operand1[1:] + ' )' if operand1.startswith('~') else operand1} , {'NOT( ' + operand2[1:] + ' )' if operand2.startswith('~') else operand2} )"
            elif operator == '^':
                pytorch_script = f"        {var_name} = XOR ( {'NOT( ' + operand1[1:] + ' )' if operand1.startswith('~') else operand1} , {'NOT( ' + operand2[1:] + ' )' if operand2.startswith('~') else operand2} )"
            class_definition += pytorch_script
            class_definition += "\n"
        else:
            if re.match(r"~?(\w+)", operand).group(1) in inputs:
                operand = operand.replace("[", "[:,")
                operand = operand.replace("]", "].unsqueeze(-1)")
            if re.match(r"~?(\w+)", operand).group(1) in outputs:
                operand = operand.replace('[', '_').replace(']', '')
            if re.match(r"~?(\w+)", var).group(1) in outputs:
                var = var.replace('[', '_').replace(']', '')
            
            pytorch_script = f"        {var} = {'NOT( ' + operand[1:] + ' )' if operand.startswith('~') else operand}"
            if operand.startswith("1'b0"):
                pytorch_script = f"        {var} = torch.zeros({inputs[0]}.shape[0], 1).to({inputs[0]}.device)"
            if operand.startswith("1'b1"):
                pytorch_script = f"        {var} = torch.ones({inputs[0]}.shape[0], 1).to({inputs[0]}.device)"

            class_definition += pytorch_script
            class_definition += "\n"

    for output, size in zip(outputs, num_outputs):
        if size > 1:
            output_list = [f"{output}_{i}" for i in range(size)]
        else:
            output_list = [f"{output}"]
        class_definition += f"        {output} = torch.stack(tuple([{', '.join(output_list)}]), dim = 1)\n"
    class_definition += "\n" \
                       f"        return [{', '.join(outputs)}]\n"
    return class_definition










