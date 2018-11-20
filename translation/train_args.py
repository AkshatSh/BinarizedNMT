import argparse
import time

def get_arg_parser():
    '''
    Create arg parse for training containing options for 
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Train Named Entity Recognition on Train Conll2003.')

    # Parser data loader options
    parser.add_argument('--clean', action='store_true', help='delete saved files')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs to train on')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='should shuffle the dataset for training')
    parser.add_argument('--batch_size', type=int, default=2, help='the batch size for specified training')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimensions for the LSTM features')
    parser.add_argument('--hidden_dim', type=int, default=1000, help='hidden dimensions of LSTM outputs for feature size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for weight update')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='directory for saved models')
    parser.add_argument('--model_name', type=str, default='model_{}'.format(time.time()), help='model_name')
    parser.add_argument('--log_dir', type=str, default='tensor_logs/', help='model_name')
    parser.add_argument('--model_path', type=str, help='the path for the saved model')
    parser.add_argument('--max_sequence_length', type=int, default=1000, help='the maximum sequence length')

    return parser