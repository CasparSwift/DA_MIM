import torch
import torch.nn.functional as F
import sys
from utils import AverageMeter
from transformers import AdamW, get_linear_schedule_with_warmup
from model import MI_loss, MI_loss2, MI_loss3


class BaseTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # self.scheduler.step()
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}
            logits = self.model(input_ids=inputs['input_ids'], masks=inputs['masks'])[0]
            index = inputs['masks'].view(-1).nonzero().view(-1)
            selected_logits = torch.index_select(logits.view(-1, self.args.n_labels), dim=0, index=index)
            selected_labels = torch.index_select(inputs['labels'].view(-1), dim=0, index=index)

            loss = self.loss_func(selected_logits, selected_labels)
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f})', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class BaseSentiTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, inputs in enumerate(self.train_loader):
            # self.scheduler.step()
            # train batch
            inputs = {k: inputs[k].to(device) for k in inputs}
            logits, senti_logits = self.model(input_ids=inputs['input_ids'], masks=inputs['masks'])
            index = inputs['masks'].view(-1).nonzero().view(-1)
            selected_logits = torch.index_select(logits.view(-1, self.args.n_labels), dim=0, index=index)
            selected_labels = torch.index_select(inputs['labels'].view(-1), dim=0, index=index)

            cls_loss = self.loss_func(selected_logits, selected_labels)
            senti_loss = self.loss_func(senti_logits, inputs['senti_label'])
            # print(inputs['labels'])
            # print(inputs['senti_label'])
            loss = cls_loss + senti_loss
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| cls_loss: {cls_loss.item():.4f} | senti_loss: {senti_loss.item():4f}', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class SemiTrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, (src_inputs, tgt_inputs) in enumerate(self.train_loader):
            # self.scheduler.step()
            # train batch
            src_inputs = {k: src_inputs[k].to(device) for k in src_inputs}
            tgt_inputs = {k: tgt_inputs[k].to(device) for k in tgt_inputs}
            logits = self.model(input_ids=src_inputs['input_ids'], masks=src_inputs['masks'])
            index = src_inputs['masks'].view(-1).nonzero().view(-1)
            selected_logits = torch.index_select(logits.view(-1, self.args.n_labels), dim=0, index=index)
            selected_labels = torch.index_select(src_inputs['labels'].view(-1), dim=0, index=index)

            cls_loss = self.loss_func(selected_logits, selected_labels)
            if epoch > 2:
                tgt_logits = self.model(input_ids=tgt_inputs['input_ids'], masks=tgt_inputs['masks'])
                max_probs = torch.max(torch.softmax(tgt_logits, dim=-1), dim=-1)[0]
                
                with torch.no_grad():
                    filter_mask = torch.tensor(max_probs > 0.5, dtype=torch.long).to(device)
                    # print(filter_mask)
                    pseudo_label_mask = filter_mask & tgt_inputs['masks']
                    pseudo_label_index = pseudo_label_mask.view(-1).nonzero().view(-1)
                    # print(pseudo_label_index)
                if torch.sum(pseudo_label_mask).cpu():
                    selected_logits_tgt = torch.index_select(tgt_logits.view(-1, self.args.n_labels), dim=0, index=pseudo_label_index)
                    pseudo_labels = torch.argmax(tgt_logits, dim=-1)
                    pseudo_labels = torch.index_select(pseudo_labels.view(-1), dim=0, index=pseudo_label_index)
                    semi_loss = self.loss_func(selected_logits_tgt, pseudo_labels)
                else:
                    semi_loss = torch.tensor(0.0).to(device)
            else:
                semi_loss = torch.tensor(0.0).to(device)
            loss = cls_loss + semi_loss
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| cls_loss: {cls_loss.item()} | semi_loss: {semi_loss.item()}', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class DATrainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.MI_loss = MI_loss2(args)

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, (src_inputs, tgt_inputs) in enumerate(self.train_loader):
            # self.scheduler.step()
            # train batch
            src_inputs = {k: src_inputs[k].to(device) for k in src_inputs}
            tgt_inputs = {k: tgt_inputs[k].to(device) for k in tgt_inputs}
            logits, _, att = self.model(input_ids=src_inputs['input_ids'], masks=src_inputs['masks'])
            index = src_inputs['masks'].view(-1).nonzero().view(-1)
            selected_logits = torch.index_select(logits.view(-1, self.args.n_labels), dim=0, index=index)
            selected_labels = torch.index_select(src_inputs['labels'].view(-1), dim=0, index=index)

            tgt_logits, _, tgt_att = self.model(input_ids=tgt_inputs['input_ids'], masks=tgt_inputs['masks'])
            # tgt_index = tgt_inputs['masks'].view(-1).nonzero().view(-1)
            # tgt_selected_logits = torch.index_select(tgt_logits.view(-1, self.args.n_labels), dim=0, index=tgt_index)

            cls_loss = self.loss_func(selected_logits, selected_labels)
            MI_loss, y_entropy = self.MI_loss(
                torch.cat([logits, tgt_logits], dim=0), 
                torch.cat([src_inputs['masks'], tgt_inputs['masks']], dim=0),
                torch.cat([att, tgt_att], dim=0),
            )
            # MI_loss, y_entropy = self.MI_loss(
            #     tgt_logits, 
            #     tgt_inputs["masks"]
            # )
            if i % self.args.balance == 0:
                loss = cls_loss + self.args.lambda_MI * MI_loss
            else:
                loss = cls_loss
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| cls_loss: {cls_loss.item():.4f} | MI_loss: {MI_loss.item():.4f}', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.debug:
                break


class DATrainer_dict(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epoch_num
        print(f'total_steps: {total_steps}', file=sys.stderr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.MI_loss = MI_loss(args)

    def train_one_epoch(self, device, epoch):
        loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.train()
        for i, (src_inputs, tgt_inputs) in enumerate(self.train_loader):
            # self.scheduler.step()
            # train batch
            src_inputs = {k: src_inputs[k].to(device) for k in src_inputs}
            tgt_inputs = {k: tgt_inputs[k].to(device) for k in tgt_inputs}
            logits = self.model(input_ids=src_inputs['input_ids'], masks=src_inputs['masks'])[0]
            index = src_inputs['masks'].view(-1).nonzero().view(-1)
            selected_logits = torch.index_select(logits.view(-1, self.args.n_labels), dim=0, index=index)
            selected_labels = torch.index_select(src_inputs['labels'].view(-1), dim=0, index=index)

            tgt_logits = self.model(input_ids=tgt_inputs['input_ids'], masks=tgt_inputs['masks'])[0]
            tgt_index = tgt_inputs['labels'].view(-1).nonzero().view(-1)
            tgt_selected_logits = torch.index_select(tgt_logits.view(-1, self.args.n_labels), dim=0, index=tgt_index)
            tgt_selected_labels = torch.index_select(tgt_inputs['labels'].view(-1), dim=0, index=tgt_index)

            tgt_index2 = tgt_inputs['masks'].view(-1).nonzero().view(-1)
            tgt_selected_logits2 = torch.index_select(tgt_logits.view(-1, self.args.n_labels), dim=0, index=tgt_index2)

            cls_loss = self.loss_func(selected_logits, selected_labels)
            # print(tgt_selected_labels)
            if tgt_selected_labels.shape[0] != 0:
                # print('add')
                cls_loss += 0.05 * self.loss_func(tgt_selected_logits, tgt_selected_labels)
            MI_loss, y_entropy = self.MI_loss(torch.cat([selected_logits, tgt_selected_logits2], dim=0))
            loss = cls_loss + self.args.lambda_MI * MI_loss
            loss_meter.update(loss.item())
            if i % self.args.logging_steps == 0:
                print(f'epoch: [{epoch}] | batch: [{i}] '\
                    f'| loss: {loss_meter.val:.4f}({loss_meter.avg:.4f}) '\
                    f'| cls_loss: {cls_loss.item():.4f} | MI_loss: {MI_loss.item():.4f}', file=sys.stderr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.debug:
                break


trainer_factory = {
    'baseline': BaseTrainer,
    'baseline_senti_cls': BaseSentiTrainer,
    'semi': SemiTrainer,
    'DA': DATrainer,
    'ner_dict': DATrainer_dict,
    'lstm_DA': DATrainer
}