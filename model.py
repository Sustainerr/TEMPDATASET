import torch
import torch.nn as nn
import re
import pickle
from tree_sitter import Language, Parser
import tree_sitter_c as tsc

class SafeCodeModel(nn.Module):
    def __init__ (self, vocabSize , embeddedDim , num_layers ,dropout, hiddenSize, nhead):
        super().__init__()


        #Loss function probably to use later
        #loss = criterion(ouput, target) in forward or main ?


        ##optimizer = optim.SGD(model.parameters(), lr=0.01)




        #Embedding layer 
        self.embedding = nn.Embedding(vocabSize, embeddedDim, padding_idx = 0)
        self.embedDropout = nn.Dropout(dropout)

        #BiGRU layer
        self.bigru = nn.GRU(
            input_size = embeddedDim,
            hidden_size = hiddenSize,
            num_layers = num_layers,
            bidirectional = True,
            batch_first = True,
            dropout = dropout
        )

        #Transformer layer
        encoderLayer = nn.TransformerEncoderLayer(
            d_model = hiddenSize * 2,
            nhead = nhead ,
            dim_feedforward = hiddenSize * 4,
            batch_first = True,
            dropout = dropout,
            activation = 'relu'
        )
        self.transformerlayer = nn.TransformerEncoder (encoderLayer , 2)

        self.classifier = nn.Sequential(
            nn.Linear(hiddenSize*2 , 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128 , 2)
        )


        pass
    
    def forward(self, x, attention_mask = None):

        x = self.embedding(x)
        x = self.embedDropout(x)
        x, _ = self.bigru(x)

        if attention_mask is not None:
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None
        
        transformer_output = self.transformerlayer(x, src_key_padding_mask=transformer_mask)
        ##pooled = torch.mean(transformer_output, dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (transformer_output * mask).sum(dim = 1)
        denom = mask.sum(dim = 1).clamp(1e-6)
        pooled = summed/denom

        x = self.classifier(pooled)
        return x


##################################################
##################################################
##################################################
##################################################
##################################################
##################################################
class TreesitterTokenizer():

    def __init__(self):
        self.PAD_TOKEN = "<PAD>" 
        self.UKW_TOKEN = "<UKW>"

        self.word2idx = {
            self.PAD_TOKEN : 0,
            self.UKW_TOKEN : 1
        }

        self.idx2word = {
            0 : self.PAD_TOKEN,
            1 : self.UKW_TOKEN
        }


        self.vocab_size = 2

        self.parser = Parser(Language(tsc.language()))


    def Tokenize (self, code):
        code_bytes = bytes(code , 'utf_8')

        tree = self.parser.parse(code_bytes)

        root_node = tree.root_node

        tokens = []

        def walk_tree(node):
            if node.child_count == 0:
                text = code_bytes[node.start_byte:node.end_byte].decode('utf_8')
                if text.strip():
                    tokens.append(text)
            else:
                for child in node.children:
                    walk_tree(child)
        
        walk_tree(root_node)
        return tokens


    def vocab (self, code_samples, min_freq):

        token_counts = {}
        for code in code_samples:

            Tokens = self.Tokenize(code)

            for token in Tokens:

                token_counts[token] = token_counts.get(token, 0) + 1
                
        for token in token_counts:
            if token_counts[token] >= min_freq and token not in self.word2idx:
                self.word2idx[token] = self.vocab_size
                self.idx2word[self.vocab_size] = token
                self.vocab_size = self.vocab_size +  1


    def encode(self, code, max_length):
        # Tokenize
        Tokens = self.Tokenize(code)
        
        # Convert to IDs
        encoded = []
        for token in Tokens:
            if token not in self.word2idx:
                encoded.append(self.word2idx[self.UKW_TOKEN])
            else:
                encoded.append(self.word2idx[token])
        
        # Safety check
        if len(encoded) == 0:
            encoded = [self.word2idx[self.UKW_TOKEN]]
        
        # Create attention mask
        attention_mask = [1] * len(encoded)
        
        # DEBUG: Print before padding
        original_len = len(encoded)
        
        # Pad or truncate
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            pad_len = max_length - len(encoded)
            encoded = encoded + [self.word2idx[self.PAD_TOKEN]] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        
        # DEBUG: Check final length
        if len(encoded) != max_length:
            print(f"❌ BUG DETECTED!")
            print(f"   Original: {original_len}, Final: {len(encoded)}, Expected: {max_length}")
            print(f"   Code preview: {str(code)[:100]}")
            raise RuntimeError(f"Padding failed! Got {len(encoded)}, expected {max_length}")
        
        if len(attention_mask) != max_length:
            print(f"❌ MASK BUG!")
            raise RuntimeError(f"Mask wrong length: {len(attention_mask)}")
        
        return encoded, attention_mask

    
    def decode(self,encoded):
        decoded = []
        for id in encoded:
            decoded.append(self.idx2word[id])

        return " ".join(decoded)
        

    def save(self, path):
        with open(path, 'wb') as f:
            save_dict = {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'PAD_TOKEN': self.PAD_TOKEN,
                'UKW_TOKEN': self.UKW_TOKEN
            }
            pickle.dump(save_dict, f)
        print(f"Tokenizer saved to {path}")
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        tokenizer = TreesitterTokenizer()
        tokenizer.word2idx = save_dict['word2idx']
        tokenizer.idx2word = save_dict['idx2word']
        tokenizer.vocab_size = save_dict['vocab_size']
        tokenizer.PAD_TOKEN = save_dict['PAD_TOKEN']
        tokenizer.UKW_TOKEN = save_dict['UKW_TOKEN']
        
        return tokenizer








if __name__ == "__main__":
    pass