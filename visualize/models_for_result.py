"""new exp"""
# vgg16_bn_half as student ver2
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    "Vanilla KD": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_104931",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_103512",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_102048",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_100621",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_095126",
# # "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_221622",  
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_10.0-20251030_154459",  
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_140555",  
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_224208",  
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251104_143420",  
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_194052",  
# # "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_1.0-20251229_184716",
    ],
    "FitNets": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_151213",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_145724",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_144246",
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_005537", 
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_014810", 
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_024046", 
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_033311", 
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_042548", 
    ],
    "Str-KD": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_015648",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_012037",

"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_213634",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_190218",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_113347",  
    ],
}


# vgg8 as student ver2
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251014_160403",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_115605",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_122500",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_125349",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_132241",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_135131",
    ],
    "Vanilla KD": [
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_115458",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_114256",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_113054",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_111851",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_110614",
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_142505", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_144737", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_151205", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_153816", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_161134", 
    ],
    "FitNets": [
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_155141",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_153916",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251230_152650",
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_051840", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_060359", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_064901", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_073407", 
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_081906", 
    ],
    "Str-KD": [
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_035404",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_032747",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_030121",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_023447",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_000018",
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_235911",
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_000018",
# "S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_235911",
    ],
}


"""ablation study on lambda_cka"""
# Effect of the CKA-based loss weight ver2
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=0$)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_104931",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_103512",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_102048",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_100621",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251230_095126",
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=10$)":[],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=100$)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_015648",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_012037",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_213634",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_190218",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_113347",  
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=200$)": [

    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=300$)": [

    ]
}


# resnet38 as student
files = {
    "Teacher": [
"resnet38x2-cifar100-trial_0-epochs_240-bs_64-20251226_140410",
    ],
    "Student": [
"resnet38-cifar100-trial_0-epochs_240-bs_64-20251230_140325",
"resnet38-cifar100-trial_0-epochs_240-bs_64-20251230_134702",
"resnet38-cifar100-trial_0-epochs_240-bs_64-20251230_133042",
"resnet38-cifar100-trial_0-epochs_240-bs_64-20251230_131420",
"resnet38-cifar100-trial_0-epochs_240-bs_64-20251226_205340",
    ],
    "Vanilla KD": [
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_1.0-20251230_125202",

"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_1.0-20251229_225634",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_1.0-20251229_223340",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251229_181129",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251229_173548",

    ],
    "FitNets": [
    ],
    "Str-KD": [
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_061812",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_051857",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_042012",

"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_160458",
"S_resnet38-T_resnet38x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_144217",
    ],
}

# resnet14 as student ver2
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_151751",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_175156",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_180721",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_145140",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_182914",
    ],
    "Vanilla KD": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_155456",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_193224",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251229_183418",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251229_175838",
    ],
    "FitNets": [
    ],
    "Str-KD": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_083708",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_081552",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251230_075504",

"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_170733",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251229_154407",
    ],
}




"""basic experiments"""
# vgg16 as student
files = {
    "Teacher": [
"vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
"vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251219_021455",
"vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251219_012030",
    ],
    "Vanilla KD": [
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251217_132152",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251217_124323",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251222_235725",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251222_230119",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251222_212714",
    ],
    "FitNets": [
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251226_045617",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251226_035750",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251226_025913",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251226_015958",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251226_010016",
    ],
    "Str-KD": [
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251219_042325",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251217_135925",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_163400",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_151202",
"S_vgg16_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_141157",
    ],
}

# vgg16_bn_half as student
# ok saved
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    "Vanilla KD": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_221622",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_10.0-20251030_154459",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_140555",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_224208",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251104_143420",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_194052",  
    ],
    "FitNets": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_005537", 
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_014810", 
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_024046", 
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_033311", 
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_042548", 
    ],
    "Str-KD": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_140650",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_194904",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_214846",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_080321",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_145309",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_115515",  
    ],
}

# vgg19_bn_half as student
# ok saved
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
"vgg19_bn_half-cifar100-trial_0-epochs_240-bs_64-20251211_090900",
"vgg19_bn_half-cifar100-trial_0-epochs_240-bs_64-20251211_094349",
"vgg19_bn_half-cifar100-trial_0-epochs_240-bs_64-20251216_164701",
"vgg19_bn_half-cifar100-trial_0-epochs_240-bs_64-20251211_102126",
"vgg19_bn_half-cifar100-trial_0-epochs_240-bs_64-20251211_105733",
    ],
    "Vanilla KD": [
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_084042",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_091156",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_093753",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_100337",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_102921",
    ],
    "FitNets": [
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_174425",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_170132",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_161944",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_153651",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_145315",
    ],
    "Str-KD": [
# "S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_120508",
# "S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_133722",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_150848",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_164053",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_181122",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_194317",
"S_vgg19_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251211_211447",
    ],
}

# vgg8 as student
# ok saved
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251014_160403",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_115605",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_122500",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_125349",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_132241",
"vgg8_bn-cifar100-trial_0-epochs_240-bs_64-20251121_135131",
    ],
    "Vanilla KD": [
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_142505", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_144737", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_151205", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_153816", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251118_161134", 
    ],
    "FitNets": [
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_051840", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_060359", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_064901", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_073407", 
"S_vgg8_bn-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251120_081906", 
    ],
    "Str-KD": [
"S_vgg8-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251119_001054", 
"S_vgg8-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251119_005355", 
"S_vgg8-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251119_013704", 
"S_vgg8-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251119_022016", 
"S_vgg8-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251119_030323", 
    ],
}

# resnet14x2 as student
# ok saved
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251219_034624",
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251219_030915",
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251223_134842",
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251223_124635",
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251223_114235",
    ],
    "Vanilla KD": [
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251217_190508",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251217_205032",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251223_145309",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251223_154017",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251223_162128",
    ],
    "FitNets": [
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_042255",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_034236",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_030332",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_022322",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_1.0-20251225_014207",
    ],
    "Str-KD": [
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251224_235257",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251224_221545",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251224_203842",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251224_190132",
"S_resnet14x2-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251224_172216",
    ],
}

# resnet14 as student
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_151751",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_175156",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_180721",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_145140",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_182914",
    ],
    "Vanilla KD": [
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_085005",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_091128",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_182245",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_184307",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_155456",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_193224",
    ],
    "FitNets": [
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251213_200128",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251213_203923",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251213_211507",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251213_215056",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251213_222641",
# betaミスってる
    ],
    "Str-KD": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_093142",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_190323",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_201930",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_103546",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251129_204735",
    ],
}

# resnet20 as student
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet20-cifar100-trial_0-epochs_240-bs_64-20251212_143229",
"resnet20-cifar100-trial_0-epochs_240-bs_64-20251212_145514",
"resnet20-cifar100-trial_0-epochs_240-bs_64-20251212_151911",
"resnet20-cifar100-trial_0-epochs_240-bs_64-20251212_154302",
"resnet20-cifar100-trial_0-epochs_240-bs_64-20251212_160656",
    ],
    "Vanilla KD": [
"S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_164007",
"S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_171137",
"S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_174121",
"S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_181111",
"S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251212_184103",
    ],
    "FitNets": [
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251212_204445",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251212_211608",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251212_214622",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251212_221640",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251212_224654"
# betaミスってる,
    ],
    "Str-KD": [
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251213_082208",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251213_094906",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251213_111217",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251213_122847",
# "S_resnet20-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251213_135429",
    ],
}

# resnet8x2 as student
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet8x2-cifar100-trial_0-epochs_240-bs_64-20251223_194435",
"resnet8x2-cifar100-trial_0-epochs_240-bs_64-20251223_192054",
"resnet8x2-cifar100-trial_0-epochs_240-bs_64-20251223_185535",
"resnet8x2-cifar100-trial_0-epochs_240-bs_64-20251223_183003",
    ],
    "Vanilla KD": [

    ],
    "FitNets": [

    ],
    "Str-KD": [

    ],
}


"""ablation study on lambda_cka"""
# Effect of the CKA-based loss weight
# ok saved
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=0$)": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_221622",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_10.0-20251030_154459",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_140555",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_224208",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251104_143420",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_194052",  
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=10$)":[],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=100$)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_194904",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_214846",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_080321",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_145309",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_115515",  
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=200$)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_200.0-20251113_110508",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_200.0-20251113_115807",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_200.0-20251113_133658",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_200.0-20251113_143440",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_200.0-20251113_153230",
    ],
    r"Str-KD ($\lambda_{\mathrm{CKA}}=300$)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_300.0-20251124_093614",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_300.0-20251124_111924",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_300.0-20251124_130216",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_300.0-20251124_144516",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_300.0-20251111_144050",

    ]
}

"""ablation study on groupe number"""
# vgg16_bn_half as student
# ok saved
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    "Vanilla KD": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_221622",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_10.0-20251030_154459",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_140555",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_224208",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251104_143420",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_194052",  
    ],
    "FitNets": [
    ],
    "Str-KD (6 groups)": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_140650",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_194904",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_214846",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_080321",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_145309",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_115515",  
    ],
    "Str-KD (8 groups)": [
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251225_132403",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251225_122109",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251225_111816",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251225_101422",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251225_090945",
    ],
}

# resnet14 as student
# ok saved
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_151751",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_175156",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_180721",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_145140",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_182914",
    ],
    "Vanilla KD": [
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_085005",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_091128",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_182245",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_184307",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_155456",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_193224",
    ],
    "FitNets": [
    ],
    "Str-KD (4 groups)": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_093142",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_190323",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_201930",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_103546",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251129_204735",
    ],
    "Str-KD (6 groups)": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_104613",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_095536",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_090455",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_081422",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_072353",
    ],
}

"""ablation study on key layers"""
# vgg16_bn_half as student
files = {
    "Teacher": [
        "vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623"
    ],
    "Student": [
# "vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251112_153023",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_090714",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_093846",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_101017",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_104150",
"vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-20251114_111323",
    ],
    "Vanilla KD": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_221622",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_10.0-20251030_154459",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_140555",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251030_224208",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251104_143420",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251031_194052",  
    ],
    "FitNets": [
    ],
    "Str-KD (All)": [
# "S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_140650",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_194904",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_214846",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251107_080321",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_145309",  
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251106_115515",  
    ],
    "Str-KD (Key)": [
# "S_vgg16_bn-T_vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251221_034148",
# "S_vgg16_bn-T_vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251221_025112",
# "S_vgg16_bn-T_vgg16_bn_half-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251220_215637",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_120439",
"S_vgg16_bn_half-T_vgg16_bn-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_113647",
    ],
}

# resnet14 as student
# ok saved
files = {
    "Teacher": [
"resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
    ],
    "Student": [
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_151751",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_175156",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_180721",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_145140",
"resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_182914",
    ],
    "Vanilla KD": [
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_085005",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_091128",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_182245",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_184307",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_155456",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_193224",
    ],
    "FitNets": [
    ],
    "Str-KD (All)": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_093142",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_190323",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_201930",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_103546",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251129_204735",
    ],
    "Str-KD (Key)": [
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_023228",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_020329",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_013513",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_005545",
"S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251222_001755",
    ],
}


""" + ablation study on groupe number + key layers """
# resnet14 as student
# files = {
#     "Teacher": [
# "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
#     ],
#     "Student": [
# "resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_151751",
# "resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_175156",
# "resnet14-cifar100-trial_0-epochs_240-bs_64-20251128_180721",
# "resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_145140",
# "resnet14-cifar100-trial_0-epochs_240-bs_64-20251129_182914",
#     ],
#     "Vanilla KD": [
# # "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_085005",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_091128",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_182245",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251128_184307",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_155456",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251129_193224",
#     ],
#     "FitNets": [
#     ],
#     "Str-KD (4 groups and all layers)": [
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_093142",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_190323",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_201930",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251128_103546",
# "S_resnet14-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251129_204735",
#     ],
#     "Str-KD (6 groups and key layers)": [
# "S_resnet14x2-T_resnet14-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251221_021421",
# "S_resnet14x2-T_resnet14-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251221_013728",
# "S_resnet14x2-T_resnet14-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251221_005920",
#     ],
# }

# # resnet8 as student
# files = {
#     "Teacher": [
# "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110",
#     ],
#     "Student": [
# "resnet8-cifar100-trial_0-epochs_240-bs_64-20251216_143635",
# "resnet8-cifar100-trial_0-epochs_240-bs_64-20251219_005349",
# "resnet8-cifar100-trial_0-epochs_240-bs_64-20251216_154108",
# "resnet8-cifar100-trial_0-epochs_240-bs_64-20251216_160002",
# "resnet8-cifar100-trial_0-epochs_240-bs_64-20251216_161848",
#     ],
#     "Vanilla KD": [
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251214_083825",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251214_091038",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251214_094102",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251214_101125",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-kd-cls_1.0-div_1.0-beta_100.0-20251216_171405",
#     ],
#     "FitNets": [
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251214_132315",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251214_135551",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251214_142640",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251214_145720",
# "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-hint-cls_1.0-div_1.0-beta_100.0-20251214_152812",
#     ],
#     "Str-KD": [
# # "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251214_162917",
# # "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251214_173034",
# # "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251214_184902",
# # "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251214_201239",
# # "S_resnet8-T_resnet14x2-cifar100-trial_0-epochs_240-bs_64-ckad-cls_1.0-div_1.0-beta_100.0-20251214_213618",
# # betaミスってる
#     ],
# }



