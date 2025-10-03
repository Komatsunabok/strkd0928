# FeatureHook class（中間特徴記録）
class FeatureHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        self.outputs.append(output)

# register_hooks関数（モデルにhookを登録して、インデックスと名前を記録）
def register_hooks(model, layer_types=None):
    """
    入力
    model：モデル
    layer_types：hookを登録する層の型（nn.BatchNorm2d, nn.Linearなど）
                Noneの場合はすべての層に登録

    出力
    hooks：登録したhookのリスト（(index, name, handle)のタプル）
        idx：モデル内の登録順に振られたインデックス。
        name：named_modules() で得られる層の名前（例：block1.1）。
        handle：register_forward_hook の戻り値。後でフックを解除したいときに使う
                （handle.remove() で解除できる）。
    feature_hook：FeatureHookのインスタンス。各層の実際の出力
    """

    hooks = []
    feature_hook = FeatureHook()

    for idx, (name, module) in enumerate(model.named_modules()):
        if (layer_types is None) or isinstance(module, layer_types):
            handle = module.register_forward_hook(feature_hook)
            hooks.append((idx, name, handle))

    return hooks, feature_hook
