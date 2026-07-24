import torch
import torch.nn as nn

from .utils import weight_initialization


class TemporalEncoder(nn.Module):
    """Encode historical sparse traffic sequences into temporal features.

    Each path node forms a token sequence from its own demand history of
    length hist_len. A Transformer encoder captures short-term temporal
    correlations, so that missing values in the latest sparse observation
    can be complemented by history.
    """

    def __init__(self, hist_len, d_model=16, num_head=2, num_encoder_layer=1):
        """Initialize temporal encoder.

        Args:
            hist_len: number of historical traffic matrices
            d_model: dimension of temporal embeddings
            num_head: number of attention heads
            num_encoder_layer: number of transformer encoder layers
        """

        super(TemporalEncoder, self).__init__()

        self.hist_len = hist_len
        self.d_model = d_model

        # project scalar demand to d_model and add learnable position
        self.input_linear = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(hist_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_head, dim_feedforward=d_model*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layer)

        # weight initialization for linear layers
        self.apply(weight_initialization)

    def forward(self, tm_seq):
        """Return temporal embeddings of the last time step.

        Args:
            tm_seq: historical sparse traffic matrices [hist_len, num_path_node]
        """

        # [num_path_node, hist_len, 1] as (batch, sequence, feature)
        # log1p compresses raw traffic scale: layer norm and softmax in
        # transformer are sensitive to large inputs and diverge otherwise
        x = torch.log1p(tm_seq.T).unsqueeze(-1)
        x = self.input_linear(x) + self.pos_embedding
        x = self.encoder(x)

        # return embeddings at the latest time step
        return x[:, -1, :]
