from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch


class IndexingTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    # def prediction_step(
    #         self,
    #         model: nn.Module,
    #         inputs: Dict[str, Union[torch.Tensor, Any]],
    #         prediction_loss_only: bool,
    #         ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     model.eval()
    #     # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
    #     with torch.no_grad():
    #         # greedy search
    #         doc_ids = model.generate(
    #             inputs['input_ids'].to(self.args.device),
    #             max_length=20,
    #             prefix_allowed_tokens_fn=self.restrict_decode_vocab)
    #     return (None, doc_ids, inputs['labels'])

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=10,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

