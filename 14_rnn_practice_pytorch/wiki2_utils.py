# import torch
import torchtext
from torchtext.data import get_tokenizer


class WikiText2Dataset:

    def __init__(self, batch_size=128, eval_batch_size=128, sequence_length=30, tokenize_func=list, device=None):
        self.tokenize_func = tokenize_func
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.symbol2idx = None
        self.idx2symbol = None
        self.TEXT = None
        self.train_txt, self.val_txt, self.test_txt = None, None, None

    def get_train_val_test_txts(self):
        self.TEXT = torchtext.data.Field(tokenize=get_tokenizer(self.tokenize_func),
                                         init_token='<sos>',
                                         eos_token='<eos>',
                                         lower=False)
        self.train_txt, self.val_txt, self.test_txt = torchtext.datasets.WikiText2.splits(self.TEXT)
        self.TEXT.build_vocab(self.train_txt)
        self.symbol2idx = self.TEXT.vocab.stoi
        self.idx2symbol = self.TEXT.vocab.itos
        return self.train_txt, self.val_txt, self.test_txt

    def _batchify(self, data_txt, batch_size):
        data = self.TEXT.numericalize([data_txt.examples[0].text])
        # Divide the dataset into batch_size parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device:
            data = data.to(self.device)
        return data

    def get_train_loader(self):
        if self.train_txt is None:
            self.get_train_val_test_txts()
        train_data = self._batchify(self.train_txt, self.batch_size)
        train_loader = TextLoader(train_data, self.sequence_length)
        return train_loader

    def get_val_loader(self):
        if self.val_txt is None:
            self.get_train_val_test_txts()
        val_data = self._batchify(self.val_txt, self.eval_batch_size)
        val_loader = TextLoader(val_data, self.sequence_length)
        return val_loader

    def get_test_loader(self):
        if self.test_txt is None:
            self.get_train_val_test_txts()
        test_data = self._batchify(self.test_txt, self.eval_batch_size)
        test_loader = TextLoader(test_data, self.sequence_length)
        return test_loader


class TextLoader:
    def __init__(self, batch_data, sequence_length=30):
        self.batch_data = batch_data
        self.sequence_length = sequence_length

    def _get_batch(self, i):
        seq_len = min(self.sequence_length, len(self.batch_data) - 1 - i)
        data = self.batch_data[i:i + seq_len]
        target = self.batch_data[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def __iter__(self):
        for i in range(0, self.batch_data.size(0) - 1, self.sequence_length):
            data, targets = self._get_batch(i)
            yield data, targets

    def __len__(self):
        return self.batch_data.size(0)
