from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.imb_factor = None  # for long-tailed cifar dataset

_C.backbone = ""
_C.resolution = 32

_C.output_dir = "magnetic_clip_vit_b16_peft"  # Directory to save the output files (like log.txt and model weights)
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.gpu = None  # assign a single gpu 
_C.num_workers = 20
_C.prec = "fp16"  # fp16, fp32, amp

_C.num_epochs = 50
_C.batch_size = 128
_C.micro_batch_size = 128  # for gradient accumulation, must be a divisor of batch size
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.weight_decay_head = 5e-4
_C.momentum = 0.9
_C.loss_type = "LA"  # "CE" / "Focal" / "LDAM" / "CB" / "GRW" / "BS" / "LA" / "LADE"

_C.classifier = "CosineClassifier"
_C.scale = 25  # for cosine classifier

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias 
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = True
_C.lora = False
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.

_C.init_head = None  # "text_feat" (only for CLIP) / "class_mean" / "1_shot" / "10_shot" / "100_shot" / "linear_probe"
_C.test_ensemble = False  # test-time ensemble
_C.expand = 24 # expand the width and height of images for test-time ensemble

_C.zero_shot = False  # zero-shot CLIP (only for CLIP)
_C.test_only = False  # load model and test
_C.test_train = False  # load model and test on the training set
_C.model_dir = None

_C.normal_label = 0

_C.patch = True # load model and test on the training set
_C.head_num = None

_C.use_randaug = False  # load model and test on the training set
_C.weighted_alpha = 1
_C.cmo = False  # load model and test on the training set

_C.weight_norm = False
_C.cmap = False
_C.attention_map = False
_C.confusion = False
_C.tsne = False

_C.train_decouple = False
