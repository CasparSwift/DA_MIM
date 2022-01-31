import argparse
import os
import torch
import torch.nn as nn
import json
import random
import numpy as np
from transformers import BertTokenizer
from dataset import make_dataloader, read_data
from model import model_factory
from trainers import trainer_factory
from tester import tester_factory, convert_tag
from utils import LSTMTokenizer

def parse_train_args():
    parser = argparse.ArgumentParser("Domain Adaptation")
    parser.add_argument("--debug", action="store_true")
    
    # model setting
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--model_type", type=str)

    # LSTM setting
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--embed_path", type=str, default='embeddings')
    
    # training and optimizer setting
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=6)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--early", type=float, default=1)
    parser.add_argument("--random_seed", type=float, default=0)
    parser.add_argument("--balance", type=int, default=1)

    # dataset setting
    parser.add_argument("--dataset", type=str, choices=['ABSA', 'NER', 'QA', 'AE'])
    parser.add_argument("--root", type=str, default='./data')
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--n_labels", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--max_sent_length", type=int, default=128)

    # DA setting
    parser.add_argument("--lambda_MI", type=float, default=0.01)
    parser.add_argument("--MI_threshold", type=float, default=0.5)

    # logging setting
    parser.add_argument("--output_dir", type=str, help="output/source-target/logging/")
    parser.add_argument("--output_name", type=str, default="run[0]")
    
    args = parser.parse_args()
    return args


def set_random_seeds(args):
    seed = float(args.random_seed)
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_train_args()
    print(args)

    set_random_seeds(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_factory[args.model_type](args)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        print('no parallel')

    model = model.to(device)

    if args.model_type != 'lstm_DA':
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    else:
        tokenizer = LSTMTokenizer(args.embed_path)

    train_loader, test_loader = make_dataloader(args, tokenizer)
    trainer = trainer_factory[args.model_type](args, model, train_loader)
    tester = tester_factory[args.model_type](args, model, test_loader)

    # if args.debug:
    #     entities_train, entities_test = set(), set()
    #     raw_inputs = read_data(args.root, args.source, 'train')
    #     for word, tags in raw_inputs:
    #         results = convert_tag(tags)
    #         for item in results:
    #             entities_train.add(' '.join(word[item[0]: item[1]+1]))
    #     raw_inputs = read_data(args.root, args.target, 'test')
    #     for word, tags in raw_inputs:
    #         results = convert_tag(tags)
    #         for item in results:
    #             entities_test.add(' '.join(word[item[0]: item[1]+1]))
    #     print(len(entities_train), len(entities_test), len(entities_test & entities_train))
    #     print(entities_train & entities_test)


    best_f1, best_f1_ae, best_precision, best_recall = 0, 0, 0, 0
    model_dir = os.path.join(args.output_dir, args.output_name)
    for epoch in range(int(args.epoch_num * args.early)):
        trainer.train_one_epoch(device, epoch + 1)
        test_result = tester.test_one_epoch(device, epoch + 1)
        precision = test_result['absa']['precision']
        recall = test_result['absa']['recall']
        f1 = test_result['absa']['F1']
        if f1 >= best_f1:
            best_f1 = f1
            best_f1_ae = test_result['ae']['F1']
            best_precision = precision
            best_recall = recall
            best_result = test_result
            torch.save(model.state_dict(), model_dir + '_model.pt')
        print(f'test_precision: {precision:.2f}, best_precision: {best_precision:.2f}')
        print(f'test_recall: {recall:.2f}, best_recall: {best_recall:.2f}')
        print(f'test_f1: {f1:.2f}, best_f1: {best_f1:.2f}, best_f1_ae: {best_f1_ae:.2f}')
        with open(model_dir + '_results.txt', 'w') as f:
            f.write(json.dumps(best_result, ensure_ascii=False))
            print(best_result)
        if args.debug:
            break

    print('----summary----')
    print(f'best_f1: {best_f1:.2f}, best_f1_ae: {best_f1_ae:.2f}')


if __name__ == '__main__':
    main()