import os
import pandas as pd

def log_cka_matrix(epoch, cka_matrix, opt, base_dir="save/cka_logs"):
    """
    CKAスコアの行列をCSVとして保存する関数

    Args:
        epoch (int): 現在のエポック数
        cka_matrix (torch.Tensor): CKAスコアの2次元テンソル
        opt: オプション設定（opt.model_name を含む）
        base_dir (str): ルート保存ディレクトリ
    """

    # モデル名に "CKA" を挿入
    model_name_with_cka = opt.model_name.replace("S", "cka_log_S", 1)

    # 保存先ディレクトリを作成
    save_dir = os.path.join(base_dir, model_name_with_cka)
    os.makedirs(save_dir, exist_ok=True) # ディレクトリが存在しない場合は作成

    # テンソルをCPUに移してnumpyに変換
    matrix_np = cka_matrix.detach().cpu().numpy()

    # PandasのDataFrameに変換
    df = pd.DataFrame(matrix_np)

    # ファイル名にエポック番号を含める（ゼロ埋め）
    filename = os.path.join(save_dir, f"cka_epoch_{epoch:03d}.csv")

    # CSVとして保存
    df.to_csv(filename, index=False)
    print(f"[CKA] Epoch {epoch}: CKA matrix saved to {filename}")