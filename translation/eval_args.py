import argparse
import train_args

def get_arg_parser() -> argparse.ArgumentParser:
    '''
    A set of parameters for evaluation
    '''
    parser = train_args.get_arg_parser()
    parser.add_argument('--load_path', type=str, help='the path of the model to test')
    parser.add_argument('--eval_train', action='store_true', help='eval on the train set')
    parser.add_argument('--eval_test', action='store_true', help='eval on the test set')
    return parser