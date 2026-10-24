from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # Attention entropyを層ごとに保存する
        entropy_by_layer = {}
        time_mass_by_layer = {}

        spectral_entropy_records = []

        # 全テストサンプル・全ヘッドの平均Attention行列を集計
        attention_sum_by_layer = {}
        attention_count_by_layer = {}

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # ===== スペクトルエントロピーを収集 =====
                if hasattr(self.model, "module"):
                    model_for_logging = self.model.module
                else:
                    model_for_logging = self.model

                if hasattr(model_for_logging, "last_spectral_entropy"):
                    spectral_entropy = (
                        model_for_logging.last_spectral_entropy
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )

                    for b in range(spectral_entropy.shape[0]):
                        for v in range(spectral_entropy.shape[1]):
                            spectral_entropy_records.append([
                                i * self.args.batch_size + b,
                                v,
                                float(spectral_entropy[b, v])
                            ])
                else:
                    if i == 0:
                        print(
                            "Warning: last_spectral_entropy が見つかりません。"
                            "モデル側の保存処理を確認してください。"
                        )
                # ===== ここまで =====

                """
                # ===== Attention確認用：ここから追加 =====
                # Attentionの取得
                if hasattr(self.model, "module"):
                    model_for_attention = self.model.module
                else:
                    model_for_attention = self.model

                attns = model_for_attention.last_attns

                n_vars = self.args.enc_in

                for layer_idx, attn in enumerate(attns):
                    if attn is None:
                        continue

                    # attn: [Batch, Heads, Tokens, Tokens]
                    A = attn.detach().float()

                    # 変数トークン同士のAttentionを抽出
                    A_var = A[..., :n_vars, :n_vars]

                    # 変数間Attentionを行ごとに再正規化
                    row_sum = A_var.sum(dim=-1, keepdim=True)
                    P = A_var / row_sum.clamp_min(1e-12)

                        # ===== 平均Attention行列の集計 =====
                    # Pの形状: [Batch, Heads, n_vars, n_vars]
                    # バッチとヘッド方向に合計する
                    batch_head_sum = P.sum(dim=(0, 1)).cpu().numpy()

                    # バッチ数 × ヘッド数
                    batch_head_count = P.shape[0] * P.shape[1]

                    if layer_idx not in attention_sum_by_layer:
                        attention_sum_by_layer[layer_idx] = np.zeros(
                            (n_vars, n_vars), dtype=np.float64
                        )
                        attention_count_by_layer[layer_idx] = 0

                    attention_sum_by_layer[layer_idx] += batch_head_sum
                    attention_count_by_layer[layer_idx] += batch_head_count
                    # ===== ここまで =====


                    # 正規化エントロピー
                    P_safe = P.clamp_min(1e-12)
                    entropy = -(P * P_safe.log()).sum(dim=-1)
                    entropy = entropy / np.log(n_vars)

                    # 各サンプルについて、全head・全変数の平均を取る
                    sample_entropy = entropy.mean(dim=(1, 2))

                    entropy_by_layer.setdefault(layer_idx, []).append(
                        sample_entropy.cpu().numpy()
                    )

                    # 変数クエリから時刻特徴量へのAttention合計
                    A_time = A[..., :n_vars, n_vars:]
                    time_mass = A_time.sum(dim=-1)
                    sample_time_mass = time_mass.mean(dim=(1, 2))

                    time_mass_by_layer.setdefault(layer_idx, []).append(
                        sample_time_mass.cpu().numpy()
                    )
                # ===== Attention確認用：ここまで =====
                """

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        import csv

        # ===== スペクトルエントロピーをCSVに保存 =====
        spectral_entropy_path = os.path.join(
            folder_path,
            "spectral_entropy_values.csv"
        )

        with open(spectral_entropy_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "window_index",
                "variable_index",
                "spectral_entropy"
            ])

            writer.writerows(spectral_entropy_records)

        print(f"Spectral entropy saved to: {spectral_entropy_path}")
        # ===== ここまで =====

        # ===== 平均Attention行列を保存 =====
        for layer_idx in sorted(attention_sum_by_layer):
            mean_attention = (
                attention_sum_by_layer[layer_idx]
                / attention_count_by_layer[layer_idx]
            )

            save_path = os.path.join(
                folder_path,
                f"mean_attention_layer{layer_idx}.npy"
            )
            np.save(save_path, mean_attention)

            print(f"Layer {layer_idx} mean Attention:")
            print(mean_attention)
            print(f"Saved to: {save_path}")
        # ===== ここまで =====

        summary_path = os.path.join(
            folder_path, "attention_entropy_summary.csv"
        )

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer",
                "mean_entropy",
                "std_entropy",
                "mean_time_attention_mass"
            ])

            for layer_idx in sorted(entropy_by_layer):
                # 全テストサンプルを結合
                ent = np.concatenate(entropy_by_layer[layer_idx])
                mass = np.concatenate(time_mass_by_layer[layer_idx])

                mean_entropy = ent.mean()
                std_entropy = ent.std()
                mean_mass = mass.mean()

                print(f"Layer {layer_idx}:")
                print(f"  Mean normalized entropy: {mean_entropy:.6f}")
                print(f"  Std normalized entropy:  {std_entropy:.6f}")
                print(f"  Mean time attention mass: {mean_mass:.6f}")

                writer.writerow([
                    layer_idx,
                    mean_entropy,
                    std_entropy,
                    mean_mass
                ])

                np.save(
                    os.path.join(
                        folder_path,
                        f"attention_entropy_layer{layer_idx}.npy"
                    ),
                    ent
                )

        print(f"Attention summary saved to: {summary_path}")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
