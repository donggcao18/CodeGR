from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import os


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
            max_samples: int=-1,
    ):
        if os.path.isdir(path_to_data):
            self.train_data = datasets.load_dataset(
                'json',
                data_files=[os.path.join(path_to_data, filename) for filename in os.listdir(path_to_data)],
                cache_dir=cache_dir
            )['train']
        else:
            self.train_data = datasets.load_dataset(
                'json',
                data_files=path_to_data,
                cache_dir=cache_dir
            )['train']
            
        if max_samples != -1:
            self.train_data = self.train_data.select(range(max_samples))

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][5:].strip() if data['text'].startswith('Code: ') else data['text']
            data['text'] = data['text'][6:].strip() if data['text'].startswith('Query: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


class GenerateDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            lang,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        
        dataset = datasets.load_dataset("json", data_files=path_to_data)["train"]
        dataset = dataset.filter(lambda example: example["text"].startswith("Code:")) # Only get data in indexing task
        for dp in dataset:
            self.data.append((dp["text_id"], "Generate docstring for this {} code: {}".format(lang, dp["text"][5:].strip())))

        # with open(path_to_data, 'r') as f:
        #     for data in f:
        #         if 'xorqa' in path_to_data:
        #             docid, passage, title = data.split('\t')
        #             for lang in self.lang2mT5.values():
        #                 self.data.append((docid, f'Generate a {lang} question for this passage: {title} {passage}'))
        #         elif 'msmarco' in path_to_data:
        #             docid, passage = data.split('\t')
        #             self.data.append((docid, f'{passage}'))
        #         else:
        #             raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(docid)


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels
