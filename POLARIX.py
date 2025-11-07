import torch
import torch.nn as nn
import torch.nn.functional as F

# H-optimus-1 output feature size.
INPUT_FEATURE_SIZE = 1536


class POLARIX(nn.Module):
    def __init__(
        self,
        precompression_layer=False,
        feature_size_comp=512,
        feature_size_attn=256,
        feature_size_comp_post=128,
        dropout=True,
        p_dropout_fc=0.5,
        p_dropout_atn=0.5,
    ):
        super(POLARIX, self).__init__()

        self.input_feature_size = INPUT_FEATURE_SIZE

        # Pre-compression layers for feature reduction
        if precompression_layer:
            self.compression_layer = nn.Sequential(
                *[
                    FC_block(
                        self.input_feature_size,
                        feature_size_comp * 4,
                        p_dropout_fc=p_dropout_fc,
                    ),
                    FC_block(
                        feature_size_comp * 4,
                        feature_size_comp * 2,
                        p_dropout_fc=p_dropout_fc,
                    ),
                    FC_block(
                        feature_size_comp * 2,
                        feature_size_comp,
                        p_dropout_fc=p_dropout_fc,
                    ),
                ]
            )

            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = self.input_feature_size

        # Attention network for MIL
        self.attention_net = AttnNet(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=1,
        )

        # Post-compression layers after attention
        self.post_compression_layer = nn.Sequential(
            *[
                FC_block(
                    dim_post_compression,
                    feature_size_comp_post,
                    p_dropout_fc=p_dropout_fc,
                )
            ]
        )

        # Classification head
        self.classifiers = nn.Linear(feature_size_comp_post, 1)

    def forward_attention(self, h):
        A_ = self.attention_net(h)
        A_raw = torch.transpose(A_, 1, 0)
        A = F.softmax(A_raw, dim=-1)
        return A_raw, A

    def forward_classification(self, m):
        logits = self.classifiers(m)
        Y_prob = torch.sigmoid(logits)
        return logits, Y_prob

    def forward(self, h):
        # H&E embedding compression
        h = self.compression_layer(h)

        # Attention-based MIL pooling
        A_raw, A = self.forward_attention(h)  # 1xN tiles

        # First-order pooling
        m = A @ h

        # Post-compression of pooled features
        m = self.post_compression_layer(m)

        # Classification
        logits, Y_prob = self.forward_classification(m)

        return logits, Y_prob, A_raw, m


class AttnNet(nn.Module):
    # Adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
    # Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()

        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(p_dropout_atn))
            self.attention_b.append(nn.Dropout(p_dropout_atn))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class FC_block(nn.Module):
    def __init__(
        self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25
    ):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x
