import argparse
import time

def get_arg_parser() -> argparse.ArgumentParser:
    '''
    Create arg parse for training containing options for 
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Train a Neural Machine Translation Model')

    # Parser data loader options
    parser.add_argument('--clean', action='store_true', help='delete saved files')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs to train on')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='should shuffle the dataset for training')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size for specified training')
    parser.add_argument('--learning_rate', type=float, default=0.0003125, help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for weight update')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='directory for saved models')
    parser.add_argument('--model_name', type=str, default='model_{}'.format(time.time()), help='model_name')
    parser.add_argument('--log_dir', type=str, default='tensor_logs/', help='model_name')
    parser.add_argument('--model_path', type=str, help='the path for the saved model')
    parser.add_argument('--max_sequence_length', type=int, default=1000, help='the maximum sequence length')
    parser.add_argument('--cuda', action='store_true', help='should run training on the GPU')
    parser.add_argument('--multi_gpu', action='store_true', help='should use multiple gpus to train')
    parser.add_argument('--save_step', type=int, default=100000, help='after how many processed examples should the model save')
    parser.add_argument('--small', action='store_true', help='use the small train set rather than the large one')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer to use')
    parser.add_argument('--model_type', type=str, default='AttentionRNN', help='the type of model to use')
    parser.add_argument('--torchtext_unk', type=int, default=2, help='minimum count for torchtext vocab')
    parser.add_argument('--torchtext_src_max_vocab', type=int, default=80000, help='the max size for the src vocab')
    parser.add_argument('--torchtext_trg_max_vocab', type=int, default=10000, help='the max size for the trg vocab')
    parser.add_argument('--torchtext_src_fix_length', type=int, default=200, help='the max token length of input')
    parser.add_argument('--log_step', type=int, default=100, help='after how many iterations to log to tensor logs')
    parser.add_argument('--should_save', type=bool, default=True, help='should create and save files')
    parser.add_argument('--dataset', type=str, default='Multi30k', help='the dataset to use (Multi30k, IWSLT, WMT)')
    parser.add_argument('--binarize', action='store_true', help='should binarize the model being used')
    parser.add_argument('--scheduler_step_size', type=int, default=5, help='the step size for the learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='scheduler scaling factor for the learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='The gradient clipping threshold')

    return parser