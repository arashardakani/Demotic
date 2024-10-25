import glob
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
import gc
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, KLDivLoss, BCEWithLogitsLoss, L1Loss
from tqdm import tqdm
from utils.comb_parser import *
from utils.seq_parser import *
from utils.verilog_parser import *

import flags
import model.circuit_comb as circuit_comb
import model.circuit_seq as circuit_seq

from utils.latency import timer


class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        '''if self.args.verbose:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Args: {self.problem_type} on {self.device}")
            logging.info(
                "\n".join([f"{k}: {v}" for k, v in self.args.__dict__.items()])
            )
        else:
            logging.basicConfig(level=logging.ERROR)'''
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.model = None
        self.loss = None
        self.optimizer = None
        self.module_name = None
        self.baseline = None
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.datasets = []
        self.num_inputs = None
        self.dataset_str = ""
        self._setup_problems()

    def _setup_problems(self):
        """Setup the problem.
        Implementation will later be extended to support other problem types. (currnetly only SAT)
            - SAT: dataset will be either (1) .cnf files in args.dataset_path or (2) PySAT PHP problems.
        """
        if self.args.problem_type == "BLIF":
            if self.args.dataset_path is None:
                raise NotImplementedError
                logging.info("No dataset found. Generating PySAT PHP problems.")
            else:
                dataset_path = os.path.join(pathlib.Path(__file__).parent.parent, self.args.dataset_path)
                self.datasets = sorted(glob.glob(dataset_path))
                self.problems = None
                self.dataset_str = self.args.dataset_path.split('/')[-1]
            self.results={}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def read_CircuitSAT_file(self, file_path):
        with open(file_path, 'r') as file:
            verilog_string = file.read()
        return verilog_string

    def _initialize_model(self, prob_id: int = 0):
        """Initialize problem-specifc model, loss, optimizer and input, target tensors
        for a given problem instance, e.g. a SAT problem (in CNF form).
        Note: must re-initialize the model for each problem instance.

        Args:
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        problem = self.read_CircuitSAT_file(self.datasets[prob_id])

        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "comb":
                bench_code = problem.replace('.', '_')
                inputs, outputs, registers = parse_combinational_module(bench_code)
                module_name = self.datasets[prob_id].split('/')[-1].replace('.bench', '').replace('.', '')
                pytorch_model = generate_combinational_pytorch_model(module_name, inputs, outputs, registers, bench_code)
                self.num_inputs = len(inputs)
                self.num_outputs = len(outputs)
            else:
                bench_code = problem.replace('.', '_')
                inputs, outputs, registers = parse_sequential_module(bench_code)
                num_outputs = len(outputs)
                module_name = self.datasets[prob_id].split('/')[-1].replace('.bench', '').replace('.', '')
                pytorch_model = generate_sequential_pytorch_model(module_name, inputs, outputs, registers, bench_code)
                self.num_inputs = len(inputs)
                self.num_outputs = num_outputs
        else:
            verilog_code = problem
            inputs, outputs, num_inputs, num_outputs = parse_verilog_module(verilog_code)
            module_name = verilog_code.split()[1]
            pytorch_model = generate_verilog_pytorch_model(module_name, inputs, outputs, num_outputs, verilog_code)
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs

        if self.args.circuit_type == "comb":
            self.model = circuit_comb.CircuitModel(
                    pytorch_model=pytorch_model,
                    inputs_str = inputs,
                    num_clk_cycles = self.args.num_clock_cycles,
                    module_name=module_name,
                    batch_size=self.args.batch_size,
                    device=self.device,
                )
        else:
            self.model = circuit_seq.CircuitModel(
                    pytorch_model=pytorch_model,
                    inputs_str = inputs,
                    num_clk_cycles = self.args.num_clock_cycles,
                    module_name=module_name,
                    batch_size=self.args.batch_size,
                    device=self.device,
                )
        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "comb":
                if module_name in ['c17']:
                    self.target = torch.ones((self.args.batch_size, 1), device=self.device) #torch.cat((d[0],d[2]), dim = -1)
                elif module_name in ['c432']:
                    self.target = torch.cat((torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device) ), dim = -1)
                elif module_name in ['c880', 'c1908', 'c3540']:
                    self.target = torch.cat((torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device) ), dim = -1)
                elif module_name in ['c499', 'c1355', 'c6288']:
                    self.target = torch.cat((torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device) ), dim = -1)
                else:
                    self.target = torch.cat((torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device)), dim = -1)
            else:
                if module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
                    self.target = torch.ones((self.args.batch_size, 1), device=self.device) #torch.cat((d[0],d[2]), dim = -1)
                elif module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
                    self.target = torch.cat((torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device) ), dim = -1)
                elif module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
                    self.target = torch.cat((torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device) ), dim = -1)
                else:
                    self.target = torch.cat((torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device), torch.zeros((self.args.batch_size, 1), device=self.device), torch.ones((self.args.batch_size, 1), device=self.device)), dim = -1)
    
        self.module_name = module_name
        self.loss = MSELoss(reduction='sum')
        self.loss_per_batch = MSELoss(reduction='none') 
        
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.results[prob_id] = {
                "prob_desc": self.datasets[prob_id].split('/')[-1],
                "num_outputs": self.num_outputs,
                "num_inputs": self.num_inputs,
            }

        self.model.to(self.device)
        self.target.to(self.device)
        self.epochs_ran = 0



    def run_back_prop_comb(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""

        target = self.target
        for epoch in train_loop:
            self.model.train()
            outputs_list = self.model()
            if self.module_name in ['c17']:
                output = outputs_list[-1] #torch.cat((d[0],d[2]), dim = -1)
            elif self.module_name in ['c432']:
                output = torch.cat(( outputs_list[0], outputs_list[-1] ), dim = -1)
            elif self.module_name in ['c880', 'c1908', 'c3540']:
                output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
            elif self.module_name in ['c499', 'c1355', 'c6288']:
                output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
            else:
                output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[31], outputs_list[63], outputs_list[-1] ), dim = -1)
            
            loss = self.loss(output, self.target) 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            
        return self.loss_per_batch(output, target)

    def run_back_prop_seq(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""

        target = self.target
        for epoch in train_loop:
            self.model.train()
            outputs_list = self.model()
            if self.module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
                if self.num_outputs > 1:
                    output = outputs_list[-1] #torch.cat((d[0],d[2]), dim = -1)
                else:
                    output = outputs_list
            elif self.module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
                output = torch.cat(( outputs_list[0], outputs_list[-1] ), dim = -1)
            elif self.module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
                output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
            else:
                output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[31], outputs_list[63], outputs_list[-1] ), dim = -1)
            
            loss = self.loss(output, self.target) 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            
        return self.loss_per_batch(output, target)
    
    def _check_solution(self):
        sol_list = []
        for param in self.model.emb.parameters_list:
            sol_list.append(param)

        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "comb":
                solutions = torch.unique( (torch.sign(torch.cat(sol_list,dim=-1))+1.)/2., dim = 0)
                new_input_list = []
                for i in range(self.num_inputs):
                    new_input_list.append(solutions[:,i].unsqueeze(-1))
                with torch.no_grad():
                    solutions_output = self.model.probabilistic_circuit_model(new_input_list)
                if self.module_name in ['c17']:
                    final_output = solutions_output[-1] #torch.cat((d[0],d[2]), dim = -1)
                elif self.module_name in ['c432']:
                    final_output = torch.cat(( solutions_output[0], solutions_output[-1] ), dim = -1)
                elif self.module_name in ['c880', 'c1908', 'c3540']:
                    final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[-1] ), dim = -1)
                elif self.module_name in ['c499', 'c1355', 'c6288']:
                    final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[-1] ), dim = -1)
                else:
                    final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[31], solutions_output[63], solutions_output[-1] ), dim = -1)
                num_unique_solutions = torch.eq(final_output, self.target[0:final_output.shape[0],:]).prod(-1).sum(-1)
            else:
                solutions = torch.unique( (torch.sign(torch.cat(sol_list,dim=-1))+1.)/2., dim = 0).flip(dims=[0])
                new_input_list = []
                for i in range(self.num_inputs):
                    new_input_list.append(solutions[:,i * self.args.num_clock_cycles : (i+1) * self.args.num_clock_cycles])
                with torch.no_grad():
                    states_pre = self.model.probabilistic_circuit_model.init_registers()
                    states = [tensor[0:solutions.shape[0], :] for tensor in states_pre]
                    for idx in range(self.args.num_clock_cycles):
                        new_input_list1 = []
                        for param in new_input_list:
                            new_input_list1.append(param[:,idx].unsqueeze(-1))
                        solutions_output, states = self.model.probabilistic_circuit_model(new_input_list1, states)
                if self.module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
                    final_output = solutions_output[-1]
                    if self.num_outputs > 1:
                        final_output = solutions_output[-1]#torch.cat((d[0],d[2]), dim = -1)
                    else:
                        final_output = solutions_output #torch.cat((d[0],d[2]), dim = -1)
                elif self.module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
                    final_output = torch.cat(( solutions_output[0], solutions_output[-1] ), dim = -1)
                elif self.module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
                    final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[-1] ), dim = -1)
                else:
                    final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[31], solutions_output[63], solutions_output[-1] ), dim = -1)
                num_unique_solutions = torch.eq(final_output, self.target[0:final_output.shape[0],:]).prod(-1).sum(-1)
                
    
    
        new_input_list = []

        

        return num_unique_solutions

    
    
    def run_model(self, prob_id: int = 0):
        solutions_found = []
        if self.args.latency_experiment:
            train_loop = range(self.args.num_steps)
            if self.args.problem_type == "BLIF":
                if self.args.circuit_type == "comb":
                    elapsed_time, losses = timer(self.run_back_prop_comb)(train_loop)
                else:
                    elapsed_time, losses = timer(self.run_back_prop_seq)(train_loop)
            logging.info("--------------------")
            logging.info("NN model solving")
            logging.info(
                f"Elapsed Time: {elapsed_time:.6f} seconds"
            )
        else:
            train_loop = (
                range(self.args.num_steps)
                if self.args.verbose
                else tqdm(range(self.args.num_steps))
            )
            if self.args.problem_type == "BLIF":
                if self.args.circuit_type == "comb":
                    losses = self.run_back_prop_comb(train_loop)
                else:
                    losses = self.run_back_prop_seq(train_loop)
        

        # if self.args.loss_fn != 'ce':
        #     losses = torch.mean(losses, dim=1)

        # sol_list = []
        # for param in self.model.emb.parameters_list:
        #     sol_list.append(param)

        solutions_found = self._check_solution()

        # solutions = torch.unique(self.model.module.get_input_weights(), dim = 0)
        # solutions_found = solutions.long().cpu().tolist()
        # solutions_found = [[(-1)**(1-sol[i]) * (i+1) for i in range(len(sol))] for sol in solutions_found]
        self.results[prob_id].update(
            {
                "model_runtime": elapsed_time,
                "model_epochs_ran": self.args.num_steps,
            }
        )
        return solutions_found

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self._initialize_model(prob_id=prob_id)
        # run NN model solving
        solutions_found = self.run_model(prob_id)
        # solver_solution = self.run_baseline(problem, prob_id)
        # is_verified = any([sol == solver_solution for sol in solutions_found])
        # if not is_verified:
        #     is_verified = self.verify_solution(problem, solutions_found)
        is_verified = solutions_found > 0
        self.results[prob_id].update(
            {
                "num_unique_solutions": solutions_found.long().cpu().tolist(),
            }
        )
        
        
        logging.info("--------------------\n")
        

    def run_all_with_baseline(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)
        if self.args.latency_experiment:
            self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}_{self.args.num_steps}"
        filename += f"_mse_{self.args.learning_rate}_{self.args.batch_size}.csv"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all_with_baseline()
