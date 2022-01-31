import argparse
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from dataset import make_dataloader, read_data
from model import model_factory
from trainers import trainer_factory
from tester import tester_factory, convert_tag


def parse_train_args():
    parser = argparse.ArgumentParser("Domain Adaptation")
    parser.add_argument("--debug", action="store_true")

    # model setting
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--vocab_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--cfg_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--model_type", type=str)
    
    # training and optimizer setting
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=6)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--early", type=float, default=1)

    # dataset setting
    parser.add_argument("--dataset", type=str, choices=['ABSA', 'NER', 'QA'])
    parser.add_argument("--root", type=str, default='./data')
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--n_labels", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--max_sent_length", type=int, default=256)

    # DA setting
    parser.add_argument("--lambda_MI", type=float, default=0.01)
    parser.add_argument("--MI_threshold", type=float, default=0.5)

    # logging setting
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="run[0]")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_train_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_factory[args.model_type](args)
    state_dict = torch.load(os.path.join(args.output_dir, 'SQuADNQ/qa/run[0]_model.pt'), map_location=device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print('no parallel')
    model = model.to(device)
    model.load_state_dict(state_dict)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir)

    train_loader, test_loader = make_dataloader(args, tokenizer)
    
    tester = tester_factory[args.model_type](args, model, test_loader)

    model_dir = os.path.join(args.output_dir, args.output_name)
    
    f1, precision, recall, f1_ae, entity_test = tester.test_one_epoch(device, 1, debug=True)

    print('----summary----')
    print(f'best_f1: {f1:.2f}, best_f1_ae: {f1_ae:.2f}')


if __name__ == '__main__':
    main()