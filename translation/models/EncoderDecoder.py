import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import numpy
from typing import Tuple

from vocab import Vocabulary
from constants import PAD_TOKEN, END_TOKEN

'''
Abstract classes for Encoder Decoder models to be used as the base for
other models
'''

DecoderOutputType = Tuple[torch.Tensor, torch.Tensor]

class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        '''
        A forward pass of the encoder model
        
        Arguments:
            src_tokens: the tokens in the source language (batch_size x max_sequence_length)
            src_lengths: the length of each sentence (batch_size), sorted such that src_lengths[0] = max_sequence_length
        
        Return:
            a torch.Tensor object for the encoded input
        '''
        pass

class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> DecoderOutputType:
        '''
        This is the forward pass for the decoder model, it relies on the correct output in (prev_output_tokens)
        since the encoder decoder models are trained with teacher enforcing

        Arguments:
            prev_output_tokens: a series of output tokens corresponding to the input, right shifted
            over so that if the orignal sequence is "a b c <end>" it becomes "<start> a b c <end>".
            encoder_out: the output from the encoder
        
        Returns:
            A tuple containg the decoder output in the first element and the attention weights in the second
            (set attention weights to None if attention is not used)

        '''
        pass

class EncoderDecoderModel(nn.Module):
    def __init__(
        self, 
        encoder: EncoderModel,
        decoder: DecoderModel,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def loss(
        self,
        predicted: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            predicted,
            expected,
            # ignore_index=self.trg_vocab.word2idx(PAD_TOKEN),
        )
    
    def generate_max(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        max_seq_len: int,
        device: str = 'cpu',
    ) -> torch.Tensor:
        batch_size = src_tokens.shape[0]
        output = torch.zeros((batch_size, max_seq_len)).long().to(device)
        encoder_out = self.encoder(src_tokens, src_lengths)
        output[:,:1] = self.trg_vocab.stoi['<sos>']
        intermediate_state = None
        for i in range(max_seq_len - 1):
            decoder_out, intermediate_state = self.decoder.forward_eval(output[:, i:i + 1], encoder_out, intermediate_state)
            topv, topi = decoder_out.max(2)
            output[:, i+1:i+2] = topi
        
        return output
    
    def slow_generate(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        max_seq_len: int,
        device: str = 'cpu',
    ) -> torch.Tensor:
        batch_size = src_tokens.shape[0]
        output = torch.zeros((batch_size, max_seq_len)).long().to(device)
        encoder_out = self.encoder(src_tokens, src_lengths)
        output[:,:1] = self.trg_vocab.stoi['<sos>']
        intermediate_state = None
        for i in range(max_seq_len - 1):
            decoder_out, _ = self.decoder(
                prev_tokens=output[:, :i + 1],
                encoder_out=encoder_out,
            )
            decoder_out = decoder_out[:, -1:]
            topv, topi = decoder_out.max(2)
            output[:, i+1:i+2] = topi
        
        return output
    
    def generate_beam(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        max_seq_len: int,
        beam_width: int,
        device: str = 'cpu',
    ) -> torch.Tensor:
        encoder_out = self.encoder(src_tokens, src_lengths)

        # (score, index of outputs, probabilities, hidden_state)
        initial_tensor = torch.zeros((1, len(self.trg_vocab))).to(device)
        initial_tensor[0][self.trg_vocab.word2idx(END_TOKEN)] = 1
        beam = [(0.0, torch.Tensor([]).to(device).long(), initial_tensor, None)]
        for i in range(max_seq_len - 1):
            new_beam = []
            for ii, curr_beam in enumerate(beam):
                score, outputs, probs, intermediate_state = curr_beam
                samples = torch.topk(probs[0], k=beam_width)[1]
                for sample in samples:
                    sample = sample.long()
                    sample_score = score + torch.log(probs[0][sample])
                    sample_tensor = torch.Tensor([sample]).to(device).unsqueeze(0).long()

                    decoder_out, intermediate_state = self.decoder(sample_tensor, encoder_out, intermediate_state)
                    # print(decoder_out.shape)
                    decoder_out = F.softmax(decoder_out, dim=2)
                    new_entry = (
                        sample_score, torch.cat([outputs, sample_tensor[0]], dim=0), decoder_out[0], intermediate_state
                    )
                    new_beam.append(new_entry)
                new_beam.sort(reverse=True, key=lambda val: val[0])
                beam = new_beam[:beam_width]
            # print('finished sequence {}'.format(i))
        # print(beam[0][1].unsqueeze(0).shape)
        return beam[0][1].unsqueeze(0)

