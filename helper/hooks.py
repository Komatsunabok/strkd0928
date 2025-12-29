import torch
import torch.nn as nn
# FeatureHook class（中間特徴記録）
class FeatureHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        self.outputs.append(output)
        
def register_hooks(
    model_name,
    model,
    *,
    # bn_suffixes=None,        # 例: ["bn2"] / ["bn1","bn2"]
    # linear_names=None,       # 例: ["fc", "classifier.6"]
    spatial_avg=False
):
    # ===== モデル別設定 =====
    if "resnet" in model_name:
        bn_suffixes = ["bn2"]
        linear_names = ["fc"]

    elif "vgg" in model_name:
        bn_suffixes = [
            name for name, m in model.named_modules()
            if isinstance(m, nn.BatchNorm2d)
        ]
        linear_names = ["classifier"]

    else:
        bn_suffixes = []
        linear_names = []

    hooks = []
    feature_hook = FeatureHook()  # ← そのまま

    EXCLUDED_LAYERS = ['layer1.0.downsample.1']

    for idx, (name, module) in enumerate(model.named_modules()):
        print(f"Checking layer: {name}")

        if name in EXCLUDED_LAYERS:
            continue

        register = False

        if isinstance(module, nn.BatchNorm2d):
            if any(name.endswith(suf) for suf in bn_suffixes):
                print(f"[HOOK BN] {name}")
                register = True

        if isinstance(module, nn.Linear):
            if name in linear_names:
                print(f"[HOOK FC] {name}")
                register = True

        if register:
            handle = module.register_forward_hook(feature_hook)
            hooks.append((idx, name, handle))

    return hooks, feature_hook

def register_hooks_conv(
    model_name,
    model,
    *,
    spatial_avg=False,
):
    """
    Register forward hooks on Conv2d layers.

    Args:
        model_name (str): name of the model (for logging).
        model (nn.Module): target model.
        spatial_avg (bool): whether to apply spatial average later (handled in FeatureHook).

    Returns:
        hooks (list): list of (idx, layer_name, hook_handle)
        feature_hook (FeatureHook): hook object storing features
    """

    hooks = []
    feature_hook = FeatureHook()

    # 除外したい層（必要に応じて追加）
    EXCLUDED_LAYERS = {
        "layer1.0.downsample.1",
    }

    for idx, (name, module) in enumerate(model.named_modules()):
        print(f"Checking layer: {name}")

        # 除外
        if name in EXCLUDED_LAYERS:
            continue

        # Conv2d のみ hook
        if isinstance(module, nn.Conv2d):
            print(f"[HOOK CONV] {name}")
            handle = module.register_forward_hook(feature_hook)
            hooks.append((idx, name, handle))

    if len(hooks) == 0:
        raise RuntimeError(f"No Conv2d layers were hooked in model {model_name}")

    return hooks, feature_hook


# # 修正後の register_hooks 関数
# def register_hooks(model, layer_types=None):
#     """
#     入力
#     model：モデル
#     layer_types：hookを登録する層の型（nn.BatchNorm2d, nn.Linearなど）
#                 Noneの場合はすべての層に登録

#     出力
#     hooks：登録したhookのリスト（(index, name, handle)のタプル）
#         idx：モデル内の登録順に振られたインデックス。
#         name：named_modules() で得られる層の名前（例：block1.1）。
#         handle：register_forward_hook の戻り値。後でフックを解除したいときに使う
#                 （handle.remove() で解除できる）。
#     feature_hook：FeatureHookのインスタンス。各層の実際の出力
#     """
#     print("register_hooks called")  # デバッグ用出力
#     hooks = []
#     feature_hook = FeatureHook()

#     # 除外したい層の名前をリスト化
#     # 今回は 'layer1.0.downsample.1' を除外
#     EXCLUDED_LAYERS = ['layer1.0.downsample.1']

#     for idx, (name, module) in enumerate(model.named_modules()):
#         print(f"Checking layer: {name}")  # デバッグ用出力
        
#         # 除外リストに含まれていたら、この層はスキップする
#         if name in EXCLUDED_LAYERS:
#             continue
        
#         # layer_typesが指定されている、または全ての層が対象の場合
#         if (layer_types is None) or isinstance(module, layer_types):
#             handle = module.register_forward_hook(feature_hook)
#             hooks.append((idx, name, handle))

#     return hooks, feature_hook

# register_hooks関数（モデルにhookを登録して、インデックスと名前を記録）
# def register_hooks(model, layer_types=None):
#     """
#     入力
#     model：モデル
#     layer_types：hookを登録する層の型（nn.BatchNorm2d, nn.Linearなど）
#                 Noneの場合はすべての層に登録

#     出力
#     hooks：登録したhookのリスト（(index, name, handle)のタプル）
#         idx：モデル内の登録順に振られたインデックス。
#         name：named_modules() で得られる層の名前（例：block1.1）。
#         handle：register_forward_hook の戻り値。後でフックを解除したいときに使う
#                 （handle.remove() で解除できる）。
#     feature_hook：FeatureHookのインスタンス。各層の実際の出力
#     """

#     hooks = []
#     feature_hook = FeatureHook()

#     for idx, (name, module) in enumerate(model.named_modules()):
#         if (layer_types is None) or isinstance(module, layer_types):
#             handle = module.register_forward_hook(feature_hook)
#             hooks.append((idx, name, handle))

#     return hooks, feature_hook
