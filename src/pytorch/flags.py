import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solving SAT/SMT/Verification problems using neural networks."
    )
    # Problem parameters
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="BLIF",
        help="Format of the CircuitSAT problem (Verilog or BLIF)",
    )
    parser.add_argument(
        "--circuit_type",
        type=str,
        default="comb",
        help="Cicuit Type (comb or seq)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--use_pgates", type=bool, default=True, help="Whether to use pgates or not"
    )

    # train(backprop) parameters
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=1e0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--momentum",
        type=str,
        default="0.0,0.9,0.99",
        help="Gradient descent momentum. Only applicable for SGD and RMSprop.",
    )
    parser.add_argument(
        "--b1",
        type=str,
        default="0.9",
        help="b1 value for Adam optimizer.",
    )
    parser.add_argument(
        "--num_clock_cycles",
        type=int,
        default=25,
        help="Number of clock cycles for sequential circuits.",
    )
    parser.add_argument(
        "--b2",
        type=str,
        default="0.99,0.999",
        help="b2 value for Adam optimizer.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Total number of gradient descent steps to perform.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=10,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="what optimizer to use"
    )
    # experiment util parameters
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether to use cpu",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=-1,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--latency_experiment",
        "-l",
        action="store_true",
        help="Whether to run latency experiment or not",
    )
    parser.add_argument(
        "--dump_solution",
        action="store_true",
        help="Whether to dump solution or not",
    )
    parser.add_argument(
        "--dump_all",
        action="store_true",
        help="Whether to dump all losses and soft assignement (embeddings)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug",
    )
    # Wandb parameters
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity (id) name"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project name"
    )
    # parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument(
        "--wandb_group", type=str, default=None, help="wandb run group name"
    )
    # parser.add_argument(
    #     "--wandb_job_type", type=str, default=None, help="wandb job type descrption"
    # )
    parser.add_argument(
        "--wandb_tags", type=str, default="", help="wandb tags, comma separated"
    )

    args = parser.parse_args()

    return args
