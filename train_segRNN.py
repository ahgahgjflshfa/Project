import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# 設定實驗參數
def main():
    args = argparse.Namespace(
        random_seed=2024,
        is_training=True,       # 記得要改
        model_id="weather_720",
        model="SegRNN",
        data="custom",
        root_path="./dataset/",
        data_path="shifts_canonical_dev_in_clean.csv",
        features="M",
        target="fact_temperature",        # 这里你需要确保目标列存在，并且改成你需要预测的特征   # TODO: 逼逼
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=720,
        label_len=0,
        pred_len=24,
        rnn_type="gru",
        dec_way="pmf",
        seg_len=12,         # 要跟著 pred_len 一起改動
        win_len=48,
        channel_id=1,
        fc_dropout=0.05,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=0,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
        embed_type=0,
        enc_in=130, # 修改为你的特征数量? 就靠報錯了吧 我不知道這到底要怎麼算...
        dec_in=116, # 修改为你的特征数量
        c_out=116,  # 修改为你的特征数量
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.5,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=1,
        itr=1,
        train_epochs=30,
        batch_size=64,
        patience=10,
        learning_rate=0.0001,
        des="test",
        loss="PEKO",
        lradj="type3",
        pct_start=0.3,
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1",
        test_flop=False
    )

    # 設定隨機種子
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # 設定實驗記錄
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.dropout,
                args.rnn_type,
                args.dec_way,
                args.seg_len,
                args.loss,
                args.des, ii)

            exp = Exp(args)  # 設定實驗
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.dropout,
            args.rnn_type,
            args.dec_way,
            args.seg_len,
            args.loss,
            args.des, ii)

        exp = Exp(args)  # 設定實驗
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()