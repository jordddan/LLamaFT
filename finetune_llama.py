"""Code for moss-sft"""

import os
import copy
import json
import torch
import logging
import argparse

import torch.distributed as dist

from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import set_seed, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

import datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),
        )

    

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
    

def train(args):

    # deepspeed needs to know your gradient accumulation steps before hand, so don't forget to pass it
    # Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=1)
    # deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 2
    accelerator = Accelerator(mixed_precision='fp16') 

    # if accelerator.is_main_process:
    #     writer = SummaryWriter(args.log_dir)
    #     writer.add_hparams(vars(args), {})

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    # transformers.LlamaForCausalLM.from_pretrained

    tokenizer = LlamaTokenizer.from_pretrained("/zecheng/llama/model/7B/hf_version")
    # tokenizer = AutoTokenizer.from_pretrained("/zecheng/llama/model/7B/hf_version", trust_remote_code=True)
    # tokenizer.eos_token_id = 106068 # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)
    # model = AutoModelForCausalLM.from_pretrained("/zecheng/llama/model/7B/hf_version", trust_remote_code=True, use_cache=False)
    model = LlamaForCausalLM.from_pretrained("/zecheng/llama/model/7B/hf_version", trust_remote_code=True, use_cache=False)
    # model.transformer.gradient_checkpointing = True
    # assert model.transformer.gradient_checkpointing is True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    dataset = datasets.load_from_disk(args.data_dir)
    train_dataset = DatasetDataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True)

    # val_dataset = SFTDataset(args.data_dir, tokenizer, data_type='val')
    # val_dataloader = DataLoader(val_dataset, batch_size=args.eval_bsz_per_gpu, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = (len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    global_step = 0
    metric = SFTMetric(device=torch.cuda.current_device())

    model.train()
    for epoch in range(args.n_epochs):
        for batch_cnt, (input_ids, labels) in enumerate(train_dataloader):
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            optimizer.zero_grad()

            output = model(input_ids=input_ids, labels=labels)
            loss = output['loss']
            if accelerator.is_main_process:
                import pdb
                pdb.set_trace()

            metric(output['logits'], labels, loss)
            acc, train_loss = metric.get_metric()

            accelerator.backward(loss)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()

            global_step += 1

            if accelerator.is_main_process:
                accelerator.print(f"epoch: {epoch}, cureent step: {batch_cnt}, total step: {len(train_dataloader)}, skip:{accelerator.optimizer_step_was_skipped}, loss:{round(train_loss, 3)}, acc:{round(acc, 3)}, length:{len(input_ids[0])}, lr:{lr_scheduler.get_last_lr()[0]}")

            # if global_step % 3 == 0 and accelerator.is_main_process:
            #     writer.add_scalar('skip', int(accelerator.optimizer_step_was_skipped), global_step=global_step)
            #     writer.add_scalar('loss', train_loss, global_step=global_step)
            #     writer.add_scalar('acc', acc, global_step=global_step)
            #     writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step=global_step)

            # if global_step % args.eval_step == 0 or global_step == 1:
            #     torch.cuda.empty_cache()
            #     model.eval() 

            #     val_metric = SFTMetric(torch.cuda.current_device())
            #     for input_ids, attention_mask, labels in val_dataloader:
            #         with torch.no_grad():
            #             output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            #         val_metric(output.logits, labels, output.loss)

            #     val_acc, val_loss = val_metric.get_metric()

            #     if accelerator.is_local_main_process:
            #         # writer.add_scalar(f'val_loss', val_loss, global_step=global_step)
            #         # writer.add_scalar(f'val_acc', val_acc, global_step=global_step)
            #         accelerator.print(f"Epoch: {epoch}, Step: {batch_cnt}, Val loss: {val_loss}, Val acc: {val_acc}")

                model.train()           

            if global_step % args.save_step == 0:
                model.save_checkpoint(args.output_dir, global_step)

    if global_step % args.save_step != 0:
        model.save_checkpoint(args.output_dir, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--model_name_or_path', default='./ckpts/moss-16B-base', type=str)
    
    # Data Args
    parser.add_argument('--data_dir', default='./Data_sample/UMLSE_Train_Tokenized', type=str)
    parser.add_argument('--output_dir', default='./ckpts/moss-16B-sft', type=str)
    parser.add_argument('--log_dir', default='./train_logs/moss-16B-sft', type=str)
    
    # Training Args
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=9e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=int)
    parser.add_argument('--n_epochs', default=2, type=int)

    # Other Args
    parser.add_argument('--save_step', default=3000, type=int)
    parser.add_argument('--eval_step', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()


    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           