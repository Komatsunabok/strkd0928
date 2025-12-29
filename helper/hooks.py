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

    if "resnet" in model_name:
        bn_suffixes = ["bn2"]                  # Block最後だけ
        linear_names = ["fc"]
    elif "vgg" in model_name:
        bn_suffixes = [name for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
        linear_names = ["classifier"]

    hooks = []
    features = {}

    bn_suffixes = bn_suffixes or []
    linear_names = linear_names or []

    def hook_fn(name):
        def fn(module, input, output):
            if spatial_avg and output.dim() == 4:
                output = output.mean(dim=(2, 3))
            features[name] = output
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if any(name.endswith(suf) for suf in bn_suffixes):
                print(f"[HOOK BN] {name}")
                hooks.append(module.register_forward_hook(hook_fn(name)))

        if isinstance(module, nn.Linear):
            if name in linear_names:
                print(f"[HOOK FC] {name}")
                hooks.append(module.register_forward_hook(hook_fn(name)))

    return hooks, features


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
