from torch import nn

class SiameseTripletNet(nn.Module):
    '''
    Siamese network for personal verification. This class accepts an embedder 
    (nn.Module) and a triplet of data instances consisting of an anchor, a
    positive example and a negative example, then returns their embeddings via
    the forward method.

    Methods:
        - __init__(embedder, device)
        - forward(anchor, pos, neg)
    
    Attributes:
        - embedder (nn.Module)
    '''
    def __init__(self, embedder):
        super(SiameseTripletNet, self).__init__()
        self.embedder = embedder
    
    def forward(self, anchor, pos, neg):
        a_embed = self.embedder(anchor)
        p_embed = self.embedder(pos)
        n_embed = self.embedder(neg)
        return a_embed, p_embed, n_embed