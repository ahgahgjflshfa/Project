import torch
import torch.nn as nn

class SegRNN(nn.Module):
    def __init__(self, configs):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len//self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len


        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

    def forward(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x) # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d

        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        # print(f"hn shape: {hn.shape}")
        # print(f"pos_emb shape: {pos_emb.shape}")

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # print(f"Shape of y: {y.shape}")
        # print(f"Shape of seq_last: {seq_last.shape}")

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last

        return y

    # def load_pretrained_weights(self, checkpoint_path):
    #     """
    #     加載預訓練權重並處理輸入層不匹配的情況。
    #     """
    #     try:
    #         checkpoint = torch.load(checkpoint_path)
    #         model_dict = self.state_dict()
    #
    #         # 遍歷預訓練模型的權重，允許處理輸入層和輸出層的維度不匹配
    #         pretrained_dict = {}
    #         for k, v in checkpoint.items():
    #             if k in model_dict:
    #                 if model_dict[k].size() == v.size():
    #                     pretrained_dict[k] = v
    #                 else:
    #                     print(
    #                         f"Skipping layer '{k}' due to size mismatch: expected {model_dict[k].size()}, got {v.size()}")
    #
    #         model_dict.update(pretrained_dict)
    #         self.load_state_dict(model_dict)
    #         print(f"成功加載預訓練權重來自 {checkpoint_path}")
    #     except Exception as e:
    #         print(f"加載預訓練權重失敗: {e}")