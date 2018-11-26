import argparse
import train_args

def get_arg_parser() -> argparse.ArgumentParser:
    '''
    A set of parameters for evaluation
    '''
    parser = train_args.get_arg_parser()
    parser.add_argument('--load_path', type=str, help='the path of the model to test')
    return parser