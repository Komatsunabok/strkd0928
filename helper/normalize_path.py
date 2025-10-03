def normalize_path(p):
    return p.replace('\\', '/')

path = r"save\cka_logs\cka_log_S-vgg13_T-vgg13_cifar10_ckad_r-1.0_a-1.0_b-400.0_0_Distill_gn-11_me-mean_red-mean_sgrp-uniform"
print(normalize_path(path))