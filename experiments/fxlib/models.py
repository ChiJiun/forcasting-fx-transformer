"""模型動物園：基準模型 + 輕量神經模型。

統一介面：
    model.fit(train_windows, val_windows)
    model.predict(windows) -> [M, H] 匯率水準預測

神經模型皆在 log-level 空間操作並做 instance normalization
（減去 context 最後一個值），等價於建模累積 log return，
可跨不同幣值尺度共用一個 global model。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------- baselines
class NaiveRW:
    name = "random_walk"

    def fit(self, train, val):
        pass

    def predict(self, w):
        H = w["tgt_level"].shape[1]
        return np.repeat(w["ctx_level"][:, -1:], H, axis=1)


class Drift:
    """Random walk with drift（用 context 內平均日變化外推）。"""
    name = "rw_drift"

    def fit(self, train, val):
        pass

    def predict(self, w):
        H = w["tgt_level"].shape[1]
        ctx = np.log(w["ctx_level"])
        drift = (ctx[:, -1] - ctx[:, 0]) / (ctx.shape[1] - 1)
        h = np.arange(1, H + 1)
        return np.exp(ctx[:, -1:] + drift[:, None] * h[None, :])


class RidgeReturns:
    """Ridge 直接多輸出回歸：預測 h=1..H 的累積 log return。

    use_feats=True 時加入 UIRP 利差、CIRP 遠期溢價等理論特徵。
    """

    def __init__(self, use_feats, n_lags=20, alpha=10.0):
        self.use_feats = use_feats
        self.n_lags = n_lags
        self.alpha = alpha
        self.name = "ridge_returns_feat" if use_feats else "ridge_returns"

    def _design(self, w):
        logc = np.log(w["ctx_level"])
        rets = np.diff(logc, axis=1)[:, -self.n_lags:]
        cols = [rets]
        if self.use_feats:
            f = w["ctx_feat"][:, -1, 1:]  # rate_diff, fwd_premium, rate_diff_chg5
            cols.append((f - self._f_mean) / self._f_std)
        return np.concatenate(cols, axis=1)

    def fit(self, train, val):
        f = train["ctx_feat"][:, -1, 1:]
        self._f_mean, self._f_std = f.mean(0), f.std(0) + 1e-12
        X = self._design(train)
        last = np.log(train["ctx_level"][:, -1:])
        y = np.log(train["tgt_level"]) - last  # cumulative log returns
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

    def predict(self, w):
        X = self._design(w)
        last = np.log(w["ctx_level"][:, -1:])
        return np.exp(last + self.model.predict(X))


# ---------------------------------------------------------------- torch nets
class _NLinearNet(nn.Module):
    def __init__(self, L, H, n_extra_feat=0):
        super().__init__()
        self.n_extra = n_extra_feat
        in_dim = L + n_extra_feat * L
        self.linear = nn.Linear(in_dim, H)

    def forward(self, x_norm, feat_norm):
        # x_norm: [B, L]（log level 減去最後值）; feat_norm: [B, L, F]
        if self.n_extra:
            x = torch.cat([x_norm, feat_norm.flatten(1)], dim=1)
        else:
            x = x_norm
        return self.linear(x)


class _DLinearNet(nn.Module):
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)

    def forward(self, x_norm, feat_norm):
        x = x_norm.unsqueeze(1)  # [B,1,L]
        pad = (self.kernel - 1) // 2
        trend = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(x, (pad, self.kernel - 1 - pad), mode="replicate"),
            self.kernel, stride=1).squeeze(1)
        season = x_norm - trend
        return self.lin_trend(trend) + self.lin_season(season)


class _PatchTSTNet(nn.Module):
    def __init__(self, L, H, n_extra_feat=0, patch_len=10, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.patch_len = patch_len
        self.n_patch = L // patch_len
        in_ch = 1 + n_extra_feat
        self.embed = nn.Linear(patch_len * in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(self.n_patch * d_model, H)

    def forward(self, x_norm, feat_norm):
        B, L = x_norm.shape
        ch = [x_norm.unsqueeze(-1)]
        if feat_norm is not None and feat_norm.shape[-1] > 0:
            ch.append(feat_norm)
        x = torch.cat(ch, dim=-1)  # [B, L, C]
        x = x[:, :self.n_patch * self.patch_len]
        x = x.reshape(B, self.n_patch, self.patch_len * x.shape[-1])
        z = self.embed(x) + self.pos
        z = self.encoder(z)
        return self.head(z.flatten(1))


class TorchForecaster:
    """共用訓練器：log-level instance norm、L1 loss、early stopping。

    normalize=False 為消融用（直接吃原始 level，示範不做平穩化的後果）。
    """

    def __init__(self, name, net_fn, use_feats=False, normalize=True,
                 seed=0, max_epochs=40, patience=6, batch_size=512, lr=1e-3):
        self.name = name
        self.net_fn = net_fn
        self.use_feats = use_feats
        self.normalize = normalize
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr

    def _prep(self, w, fit_scaler=False):
        level = torch.tensor(w["ctx_level"], dtype=torch.float32)
        x = torch.log(level)
        last = x[:, -1:]
        x_in = (x - last) if self.normalize else x
        if self.use_feats:
            f = torch.tensor(w["ctx_feat"], dtype=torch.float32)
            if fit_scaler:
                self._fm = f.mean(dim=(0, 1))
                self._fs = f.std(dim=(0, 1)) + 1e-12
            f = (f - self._fm) / self._fs
        else:
            f = torch.zeros(x.shape[0], x.shape[1], 0)
        return x_in, f, last

    def _target(self, w, last):
        y = torch.log(torch.tensor(w["tgt_level"], dtype=torch.float32))
        return (y - last) if self.normalize else y

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        L = train["ctx_level"].shape[1]
        H = train["tgt_level"].shape[1]
        self.net = self.net_fn(L, H)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.L1Loss()

        xt, ft, lt = self._prep(train, fit_scaler=self.use_feats)
        yt = self._target(train, lt)
        xv, fv, lv = self._prep(val)
        yv = self._target(val, lv)

        n = xt.shape[0]
        best_val, best_state, bad = np.inf, None, 0
        for _epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.net(xt[idx], ft[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(xv, fv), yv).item()
            if vl < best_val - 1e-6:
                best_val, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, w):
        x, f, last = self._prep(w)
        self.net.eval()
        with torch.no_grad():
            out = self.net(x, f)
        y = (out + last) if self.normalize else out
        return np.exp(y.numpy())


def build_models(seed=0):
    """完整實驗矩陣。"""
    return [
        NaiveRW(),
        Drift(),
        RidgeReturns(use_feats=False),
        RidgeReturns(use_feats=True),
        TorchForecaster("nlinear", lambda L, H: _NLinearNet(L, H), seed=seed),
        TorchForecaster("nlinear_feat", lambda L, H: _NLinearNet(L, H, n_extra_feat=4),
                        use_feats=True, seed=seed),
        TorchForecaster("dlinear", lambda L, H: _DLinearNet(L, H), seed=seed),
        TorchForecaster("dlinear_raw_level", lambda L, H: _DLinearNet(L, H),
                        normalize=False, seed=seed),
        TorchForecaster("patchtst", lambda L, H: _PatchTSTNet(L, H), seed=seed),
        TorchForecaster("patchtst_feat", lambda L, H: _PatchTSTNet(L, H, n_extra_feat=4),
                        use_feats=True, seed=seed),
    ]
