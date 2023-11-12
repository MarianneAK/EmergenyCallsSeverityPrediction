import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from transformers import get_linear_schedule_with_warmup, CamembertTokenizer, CamembertForSequenceClassification

from sklearn.model_selection import train_test_split
import pickle
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import os
import random
import torch.optim as optim
import statistics

root_dir = '/Home/Users/mabi_kanaan/WORK/eop_clean/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MAX_LEN = 384 

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_data(tokenizer, my_seed=42):

    X = []
    y = []

    '''
        This block of code is supposed to fill X and y.
        X -> emergency call texts
        y -> severity labels

        Currently unavailable due to it being sensitive code.
        TODO: re-implement it in a non-sensitive way
    '''

    train_text, test_text, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=my_seed, stratify=y)
    
    with open(root_dir + "train.tsv", 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for i in range(0, len(train_text)):
            tsv_writer.writerow([train_text[i], train_labels[i]])
            
    with open(root_dir + "test.tsv", 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for i in range(0, len(test_text)):
            tsv_writer.writerow([test_text[i], test_labels[i]])


    TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=tokenizer, use_vocab=False, lower=False,
                            include_lengths=True, batch_first=True, pad_token=1)
    LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

    train_data, test_data = torchtext.legacy.data.TabularDataset.splits(
        path=root_dir, train='train.tsv', test='test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

    train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_data, test_data),
                                                        batch_sizes=(BATCH_SIZE, 1), 
                                                       repeat=False, sort=False)
    return train_iter, test_iter


def get_bert_classifier():
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', output_hidden_states=True, num_labels=2)
    bert_classifier = CamembertClassifier(tokenizer, model)
    return bert_classifier, tokenizer


class CamembertClassifier(nn.Module):
    def __init__(self, tokenizer, model):
        super(CamembertClassifier, self).__init__()
        self.layer_size = 512
        self.tokenizer = tokenizer
        self.bert = model
        # BERT hidden state size is 768, class number is 2
        self.linear = nn.Linear(768, 2)

        # initialing weights and bias
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
        
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, 768)
    
    def bert_tokenizer(self, text):
        return self.tokenizer.encode(text, return_tensors='pt', max_length=MAX_LEN, padding='max_length', truncation=True, add_special_tokens=True)[0]

    def forward(self, input_ids):
        # get last_hidden_state
        hidden_states = self.bert(input_ids).hidden_states

        vec = self._get_cls_vec(hidden_states[-1])
        # vec2 = self._get_cls_vec(hidden_states[-2])
        # vec3 = self._get_cls_vec(hidden_states[-3])
        # vec4 = self._get_cls_vec(hidden_states[-4])

        out = self.linear(vec)
        return F.log_softmax(out)

def train(bert_classifier, train_iter, test_iter,tokenizer):
    n_epochs = 12

    # fine tuning
    for param in bert_classifier.parameters():
        param.requires_grad = False

    # BERT last 1 layer; ON, last 3 layers commented
    for param in bert_classifier.bert.roberta.encoder.layer[-1].parameters():
        param.requires_grad = True

    # for param in bert_classifier.bert.roberta.encoder.layer[-2].parameters():
    #     param.requires_grad = True

    # for param in bert_classifier.bert.roberta.encoder.layer[-3].parameters():
    #     param.requires_grad = True

    # for param in bert_classifier.bert.roberta.encoder.layer[-4].parameters():
    #     param.requires_grad = True

    for param in bert_classifier.linear.parameters():
        param.requires_grad = True


    optimizer = optim.Adam([
        {'params': bert_classifier.bert.roberta.encoder.layer[-1].parameters(), 'lr': 5e-5},
        # {'params': bert_classifier.bert.roberta.encoder.layer[-2].parameters(), 'lr': 5e-5},
        # {'params': bert_classifier.bert.roberta.encoder.layer[-3].parameters(), 'lr': 5e-5},
        # {'params': bert_classifier.bert.roberta.encoder.layer[-4].parameters(), 'lr': 5e-5},
        {'params': bert_classifier.linear.parameters(), 'lr': 3e-5, 'eps': 1e-07}
    ])

    total_steps = len(train_iter) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = int(0.1 * total_steps),
                                    num_training_steps = total_steps)

    loss_function = nn.NLLLoss()


    # send network to GPU
    bert_classifier.to(device)
    bert_classifier = bert_classifier.float()
    best_acc = 0
    best_pr = 0
    best_rec = 0
    best_f1 = 0

    for epoch in range(n_epochs):
        all_loss = 0
        for idx, batch in enumerate(train_iter):
            batch_loss = 0

            bert_classifier.zero_grad()
            input_ids = batch.Text[0].to(device)
            label_ids = batch.Label.to(device)
            out = bert_classifier(input_ids)

            batch_loss = loss_function(out, label_ids)
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            all_loss += batch_loss.item()

        acc, pr, rec, f1 = evaluate(test_iter, bert_classifier)
        if acc > best_acc:
            best_acc = acc
            best_pr = pr
            best_rec = rec
            best_f1 = f1
        print("epoch", epoch, "\t" , "loss", all_loss, " acc ", acc)

    return best_acc, best_pr, best_rec, best_f1

def evaluate(test_iter, bert_classifier):
    answer = []
    prediction = []

    with torch.no_grad():
        for batch in test_iter:

            text_tensor = batch.Text[0].to(device)
            label_tensor = batch.Label.to(device)
    
            score = bert_classifier(text_tensor)
            _, pred = torch.max(score, 1)
            prediction += list(pred.cpu().numpy())
            answer += list(label_tensor.cpu().numpy())

    print("\nSeverity prediction confusion matrix")
    print(confusion_matrix(answer, prediction))

    return accuracy_score(answer, prediction), precision_score(answer, prediction, average='macro'),recall_score(answer, prediction, average='macro'), f1_score(answer, prediction, average='macro')


if __name__ == '__main__':

    seeds = [777, 123, 42, 555, 666, 888, 111, 222, 100, 10]
    accuracies = []
    recalls = []
    precisions = []
    f1s = []

    for my_seed in seeds:
        seed_everything(my_seed)
        bert_classifier, tokenizer = get_bert_classifier()
        train_iter, test_iter = get_data(tokenizer, my_seed=my_seed)
        acc,rec,pre,f1 = train(bert_classifier, train_iter, test_iter, tokenizer)
        accuracies.append(acc)
        recalls.append(rec)
        precisions.append(pre)
        f1s.append(f1)

print("Mean accuracy ", str(statistics.fmean(accuracies)))
print("Mean recall ", str(statistics.fmean(recalls)))
print("Mean precision ", str(statistics.fmean(precisions)))
print("Mean f1 ", str(statistics.fmean(f1s)))
