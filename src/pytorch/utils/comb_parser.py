import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import networkx as nx
import re



def extract_variables(expr, inputs, registers):
    # Regular expression pattern to match variables inside parentheses
    pattern = r'\((.*?)\)'
    # Find all matches of the pattern in the expression
    matches = re.findall(pattern, expr)
    # Split each match by commas and strip to get individual variables
    variables = [var.strip() for match in matches for var in match.split(',')]
    #variables = [var for var in variables if var not in inputs]
    #variables = [var for var in variables if not var.startswith('self.')]
    return variables

def refine(text, registers, inputs):
    lines = text.strip().split('\n')
    filtered_lines = [line for line in lines if not line.startswith('#') and 'INPUT' not in line and 'OUTPUT' not in line and 'DFF' not in line]
    registered_lines = [line for line in lines if 'DFF' in line]
    
    modified_lines = []
    states = []
    for line in registered_lines:
        variable, value = line.split('=')        
        modified_variable = f"{variable.strip()}"        
        value = value.split('(')[1].split(')')[0]
        states.append(value)
        modified_line = f"{modified_variable} = {value}"
        modified_lines.append(modified_line)
    
    #filtered_lines = [line for line in filtered_lines if 'INPUT' not in line and 'OUTPUT' not in line]
    filtered_string = '\n'.join(filtered_lines)
    non_empty_lines = [line for line in filtered_string.split('\n') if line.strip() != '']
    filtered_string = '\n'.join(non_empty_lines)
    filtered_string = filtered_string.replace(' not(', ' NOT(')
    filtered_string = filtered_string.replace(' and(', ' AND(')
    filtered_string = filtered_string.replace(' or(', ' OR(')
    filtered_string = filtered_string.replace(' nand(', ' NAND(')
    filtered_string = filtered_string.replace(' nor(', ' NOR(')
    filtered_string = filtered_string.replace(' xor(', ' XOR(')
    filtered_string = filtered_string.replace(' xnor(', ' XNOR(')
    filtered_string = filtered_string.replace(' buf(', ' BUF(')
    
    '''for variable in registers:
        if variable in filtered_string:
            #filtered_string = filtered_string.replace(variable, f'self.{variable}')
            filtered_string = re.sub(rf'\b{variable}\b', f'self.{variable}', filtered_string)'''
    
    # for line in filtered_string.strip().split('\n'):
    #     variables = extract_variables(line)
    
    dependencies = {}
    expressions = {}
    # Extract variable names and their dependencies
    for line in filtered_string.strip().split('\n'):
        parts = line.split('=')
        variable = parts[0].strip()
        expr = parts[1].strip() if len(parts) > 1 else ""
        dependencies[variable] = extract_variables(expr, inputs, registers)
        expressions[variable] = parts[1].strip()
        
    Graph = nx.DiGraph()
    for variable, deps in dependencies.items():
        for dep in deps:
            Graph.add_edge(dep, variable)
            '''print(dep, variable)
            sdads'''
    # Topological sorting of statements
    sorted_vars = list(nx.topological_sort(Graph))
    cnt = 0
    for var in sorted_vars:
        if var not in inputs:
            cnt += 1 
    
    
    rearranged_statements = [f"{var} = {expressions[var]}" for var in sorted_vars if var not in inputs]
    modified_input_string = '\n'.join(rearranged_statements)
    
    
    return modified_input_string.strip().split('\n'), states #modified_lines



def parse_combinational_module(verilog_code):
    lines = verilog_code.split('\n')
    inputs = []
    outputs = []
    registers = []
    
    for line in lines:
        
        if 'INPUT' in line:
            if line.startswith('INPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                inputs.append(variable)
        elif 'OUTPUT' in line:
            if line.startswith('OUTPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                outputs.append(variable)
        elif 'DFF' in line:
            variable = line.split('=')[0].strip().replace('.', '')
            registers.append(variable)
    return inputs, outputs, registers

def generate_combinational_pytorch_model(module_name, inputs, outputs, registers, assignments):
    class_name = module_name.lower()
    class_definition = f"class {class_name}(nn.Module):\n" \
                       f"    def __init__(self, batch_size, device):\n" \
                       f"        super().__init__()\n"
    class_definition += f"        self.batch_size = batch_size\n"
    class_definition += f"        self.device = device\n"
    '''for register in registers:
        class_definition += f"        self.{register} = torch.zeros((batch_size, 1), device = device)\n"
    class_definition += "\n"
    class_definition += f"    def reset_registers(self):\n"
    for register in registers:
        class_definition += f"        self.{register} = self.{register} * 0\n"
    class_definition += "\n"
    class_definition += f"    def detach_registers(self):\n"
    for register in registers:
        class_definition += f"        self.{register} = self.{register}.detach()\n"'''
    class_definition += "\n"
    class_definition += f"    def forward(self, inputs):\n"
    class_definition += f"        {', '.join(inputs)} = inputs\n"
    logics_assignments, registers_assignments = refine(assignments, registers, inputs)
    
    for assignment in logics_assignments:
        class_definition += f"        {assignment}\n"
    '''for assignment in registers_assignments:
        class_definition += f"        {assignment}\n"'''
    class_definition += f"        outputs = {', '.join(outputs)}\n"
    class_definition += "\n" \
                       f"        return outputs\n"
    return class_definition