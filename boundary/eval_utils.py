import vocab
from config import *


class UsableTransformer:
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        self.vocab = vocab.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        map_location = CUDA_DEVICE if USE_CUDA else torch.device("cpu")
        self.model = torch.load(model_path, map_location=map_location, weights_only=False)
        if USE_CUDA:
            self.model = self.model.cuda(CUDA_DEVICE)
        else:
            self.model = self.model.cpu()

    def _encode_single(self, instructions):
        segment_label = []
        sequence = []
        for ins in instructions:
            l = (len(ins.split(' ')) + 2) * [1]
            s = self.vocab.to_seq(ins)
            s = [3] + s + [2]
            if len(l) > 20:
                segment_label.append(l[:20])
            else:
                segment_label.append(l + [0] * (20 - len(l)))
            if len(s) > 20:
                sequence.append(s[:20])
            else:
                sequence.append(s + [0] * (20 - len(s)))
        return segment_label, sequence

    def encode(self, batch_text):
        # Accept a batch of sequences (list[list[str]]) or a single sequence (list[str]).
        if len(batch_text) == 0:
            return torch.empty(0, 0, 0)

        if isinstance(batch_text[0], str):
            batch_text = [batch_text]

        batch_segment = []
        batch_sequence = []
        seq_len = len(batch_text[0])
        for sample in batch_text:
            seg, seq = self._encode_single(sample)
            batch_segment.extend(seg)
            batch_sequence.extend(seq)

        segment_label = torch.LongTensor(batch_segment)
        sequence = torch.LongTensor(batch_sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE, non_blocking=True)
            segment_label = segment_label.cuda(CUDA_DEVICE, non_blocking=True)

        with torch.no_grad():
            encoded = self.model.forward(sequence, segment_label)
            # encoded: (batch*seq_len, token_len, hidden)
            pooled = torch.mean(encoded.detach(), dim=1)
            result = pooled.view(len(batch_text), seq_len, -1)

        # Ensure batch-first layout.
        if result.shape[0] != len(batch_text) and result.ndim == 3 and result.shape[1] == len(batch_text):
            result = result.permute(1, 0, 2).contiguous()

        return result
