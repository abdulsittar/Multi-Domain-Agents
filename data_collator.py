import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollatorForCausalLM:
    def __init__(self, tokenizer, padding_value=None, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.padding_value)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

        attention_mask = input_ids_padded.ne(self.padding_value).long()

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }
