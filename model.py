import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import BertConfig, BertModel
import numpy as np
import os
import math


class GradReverse(Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

# base MIM
class MI_loss(nn.Module):
    def __init__(self, args):
        super(MI_loss, self).__init__()
        self.MI_threshold = args.MI_threshold

    def forward(self, logits, masks):
        n_labels = logits.shape[-1]
        index = masks.view(-1).nonzero().view(-1)
        selected_logits = torch.index_select(logits.view(-1, n_labels), dim=0, index=index)
        p = F.softmax(selected_logits, dim=-1)
        # BIO tagging
        if p.shape[1] == 3:
            p = torch.stack([p[:, 0], torch.sum(p[:, 1:], dim=-1)]).T
        condi_entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
        y_dis = torch.mean(p, dim=0)
        y_entropy = (-y_dis * torch.log(y_dis)).sum()
        if y_entropy.item() < self.MI_threshold:
            return -y_entropy + condi_entropy, y_entropy
        else:
            return condi_entropy, y_entropy

# dynamic threshold
class MI_loss2(nn.Module):
    def __init__(self, args):
        super(MI_loss2, self).__init__()
        self.MI_threshold = args.MI_threshold

    def softmask(self, score, mask):
        score_exp = torch.mul(torch.exp(score), mask)
        sumx = torch.sum(score_exp, dim=-1, keepdim=True)
        return score_exp / (sumx + 1e-5)

    def forward(self, logits, masks, att):
        # [bsz * seq * label]
        p = F.softmax(logits, dim=-1)
        # BIO tagging
        if p.shape[-1] == 3:
            p = torch.stack([p[:, 0], torch.sum(p[:, 1:], dim=-1)]).T

        bsz, seq_len, n_labels = p.shape
        real_lens = torch.sum(masks, dim=-1)
        condi_entropy = -torch.sum(p * torch.log(p), dim=-1)
        condi_entropy = torch.sum(condi_entropy * masks) / torch.sum(real_lens)

        # att = att.view(bsz, seq_len)
        # att = self.softmask(att, masks)
        # att = att.unsqueeze(-1)

        u = torch.rand_like(att)
        att = torch.sigmoid((att+torch.log(u)-torch.log(1-u)) / 0.5)
        # att = torch.sigmoid(att)
        # print(att)
        repeat_att = torch.cat([att, att], dim=-1)

        masks = masks.unsqueeze(-1)
        repeat_masks = torch.cat([masks, masks], dim=-1)

        # p_sum = torch.sum((p * repeat_masks).view(-1, p.shape[-1]), dim=0)
        # y_dis = p_sum / torch.sum(real_lens)

        p = p * repeat_masks
        p = p * repeat_att

        p_sum = torch.sum(p * repeat_masks, dim=1)
        # print(p_sum.shape)
        y_dis = torch.div(p_sum.T, real_lens).T
        y_entropy = torch.sum(-y_dis * torch.log(y_dis), dim=-1)

        # print(threshold)
        y_entropy_mask = (y_entropy < self.MI_threshold).long()
        # print(y_dis)
        # print(y_entropy)
        # print(y_entropy_mask)
        y_entropy_loss = (y_entropy * y_entropy_mask).sum() / (y_entropy_mask.sum()+1e-6)
        loss = -y_entropy_loss + condi_entropy
        return loss, y_entropy_loss


# fine-grained
class MI_loss3(nn.Module):
    def __init__(self, args):
        super(MI_loss3, self).__init__()
        self.MI_threshold = args.MI_threshold

    def forward(self, logits, masks):
        # [bsz * seq * label]
        p = F.softmax(logits, dim=-1)
        # BIO tagging
        if p.shape[-1] == 3:
            p = torch.stack([p[:, 0], torch.sum(p[:, 1:], dim=-1)]).T

        bsz, seq_len, n_labels = p.shape
        real_lens = torch.sum(masks, dim=-1)
        condi_entropy = -torch.sum(p * torch.log(p), dim=-1)
        condi_entropy = torch.sum(condi_entropy * masks) / torch.sum(real_lens)

        masks = masks.unsqueeze(-1)
        repeat_masks = torch.cat([masks, masks], dim=-1)

        # p_sum = torch.sum((p * repeat_masks).view(-1, p.shape[-1]), dim=0)
        # y_dis = p_sum / torch.sum(real_lens)

        p_sum = torch.sum(p * repeat_masks, dim=1)
        # print(p_sum.shape)
        y_dis = torch.div(p_sum.T, real_lens).T
        y_entropy = torch.sum(-y_dis * torch.log(y_dis), dim=-1)
        y_entropy_mask = (y_entropy < self.MI_threshold).long()
        # print(y_dis)
        # print(y_entropy)
        # print(y_entropy_mask)
        y_entropy_loss = (y_entropy * y_entropy_mask).sum() / (y_entropy_mask.sum()+1e-6)
        loss = -y_entropy_loss + condi_entropy
        return loss, y_entropy_loss


class BaseBert(nn.Module):
    def __init__(self, args):
        super(BaseBert, self).__init__()
        print(f"Initializing main bert model from {args.model_dir}...")
        model_config = BertConfig.from_pretrained(args.model_dir)
        self.bert_model = BertModel.from_pretrained(args.model_dir, config=model_config)
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size//2),
            nn.ReLU(),
            nn.Linear(args.hidden_size//2, args.hidden_size//2),
            nn.ReLU(),
            nn.Linear(args.hidden_size//2, args.n_labels)
        )
        self.attention = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            # nn.ReLU(),
            # nn.Linear(args.hidden_size//2, 1),
        )

    def forward(self, input_ids, masks):
        output = self.bert_model(input_ids, masks)[0]
        logits = self.classifier(output)
        att = self.attention(output)
        return logits, output, att


class BaseBert_with_senti_cls(nn.Module):
    def __init__(self, args):
        super(BaseBert_with_senti_cls, self).__init__()
        print("Initializing main bert model...")
        model_config = BertConfig.from_pretrained(args.model_dir)
        self.bert_model = BertModel.from_pretrained(args.model_dir, config=model_config)
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_labels)
        )
        self.senti_classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_labels)    
        )

    def forward(self, input_ids, masks):
        output = self.bert_model(input_ids, masks)[0]
        logits = self.classifier(output)
        senti_logits = self.senti_classifier(output[:, 0, :])
        return logits, senti_logits


class BaseBert_DANN(nn.Module):
    def __init__(self, args):
        super(BaseBert_DANN, self).__init__()
        print("Initializing main bert model...")
        model_config = BertConfig.from_pretrained(args.model_dir)
        self.bert_model = BertModel.from_pretrained(args.model_dir, config=model_config)
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_labels)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 2)
        )

    def forward(self, input_ids, masks, constant):
        output = self.bert_model(input_ids, masks)[0]
        logits = self.classifier(output)
        senti_logits = self.senti_head(output[:, 0, :])
        domain_preds = self.domain_classifier(
            GradReverse.grad_reverse(output[:, 0, :], constant))
        return logits, senti_logits


class DECNN_CONV(nn.Module):
    def __init__(self, input_dim=100):
        super(DECNN_CONV, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_dim, 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(input_dim, 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(0.5)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)

    def forward(self, inputs):
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(inputs), self.conv2(inputs)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)
        return x_conv


class BaseLSTM(nn.Module):
    def __init__(self, args):
        super(BaseLSTM, self).__init__()
        self.weight = np.load(os.path.join(args.embed_path, 'cross_embedding.npy'))
        # self.rnn = torch.nn.LSTM(
        #     input_size=self.weight.shape[1],
        #     hidden_size=args.lstm_hidden_size,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.25
        # )
        self.cnn = DECNN_CONV()
        self.embedding = torch.nn.Embedding(self.weight.shape[0], self.weight.shape[1])
        self.input_dropout = nn.Dropout(0.4)
        self.embedding.weight.data.copy_(torch.from_numpy(np.array(self.weight)))
        self.embedding.requires_grad = True

        self.classifier = nn.Linear(256, args.n_labels)
        self.init()

    def init(self):
        for name, p in self.cnn.named_parameters():
            if 'bert' not in name:
                if p.requires_grad:
                    if len(p.shape) > 1:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def forward(self, input_ids, masks):
        # output, (h_n, h_c) = self.rnn(self.embedding(input_ids), None)
        output = self.cnn(self.embedding(input_ids).transpose(1, 2))
        logits = self.classifier(output)
        # logits = torch.clamp(logits, 1e-5, 1.)
        return logits, output


model_factory = {
    'baseline': BaseBert,
    'baseline_senti_cls': BaseBert_with_senti_cls,
    'semi': BaseBert,
    'DA': BaseBert,
    'ner_dict': BaseBert,
    'lstm_DA': BaseLSTM
}