import numpy as np
from torch import nn

# クラス定義
class Autoencoder(nn.Module):
    def __init__(self, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(52, z_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 52),
            nn.ReLU(True)
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

def AE_T2RE(X, z_hat, S_z):
    """
    T2 統計量とRE を計算します

    パラメータ
    ----------
    X: 監視対象データ
    z_hat: 正常データでの中間層の値の平均値
    S_z: 正常データでの中間層の値の共分散行列

    戻り値
    -------
    T2_AE: T2 統計量
    RE_AE: RE
    """

    xhat_tensor, z_tensor = model.forward(X)
    z = z_tensor.detach().numpy()
    xhat = xhat_tensor.detach().numpy()

    T2_AE = np.empty(len(X))
    RE_AE = np.empty(len(X))

    for i in range(len(z)):
        z_vec = np.reshape(z[i], (len(z[i]), 1))

        T2 = (z_vec - z_hat).T @ np.linalg.inv(S_z)@(z_vec - z_hat)
        RE = (X[i] - xhat [i])**2

        T2_AE [i] = T2[0]
        RE_AE [i] = Q[0]

    return T2_AE, RE_AE