# Models

Here are all the PyTorch files for our machine translation models

## Contributing

To add a new model, the following things must be supported, to plug in with anything else.

### Super classes

We have implemented the following abstract classes
* EncoderModel (EncoderDecoder.py)
* DecoderModel (EncoderDecoder.py)
* EncoderDecoderModel(EncoderDecoder.py)

When creating a new encoder decoder model, make sure the encoder model subclasses the encoder model, and the decoder model subclasses the decoder model. The training pipeline set up uses teacher enforced learning, but we should also look into having some teacher student ratio for learning.

### EncoderModel

When writing your encoder, you must create a subclass of `EncoderModel` and look into implementing the following abstract method:

```python
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
    return None
```


## DecoderModel

Similar to the encoder model, when writing the decoder model, `DecoderModel` must be inhereited and the following abstract method must be implemented.

```python
DecoderOutputType = Tuple[torch.Tensor, torch.Tensor]
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
        return None
```

## EncoderDecoderModel

Once the encoder and decoder have been implemented we have implemented an encoder decoder network, that encapsulates the encoder and decoder implemented. If you want to see the implementation take a look at `EncoderDecoderModel` in `EncoderDecoder.py`.

In order for this to work you must implement the following method in your file:

```python
def build_model(
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
    # encoder arguments
    # decoder arguments
) -> nn.Module:
    encoder = YourEncoder(
       # encoder_args...
    )

    decoder = YourDecoder(
        # decoder_args...
    )

    return EncoderDecoderModel(
        encoder,
        decoder,
        en_vocab,
        fr_vocab,
    )

def add_args(parser: argparse.ArgumentParser) -> None:
    '''
    Add all the hyper parameters as arguments to the parser
    '''
    return None
```

## Notes

In order to get the QRNN model working in env/lib/python3.6/site-packages/pynvrtc/interface.py line 54 was slightly modified. There is some issue where the input, s, has already been converted to byte form so it does not have the encode() method. Therefore the the code is modified so that s is just returned.
