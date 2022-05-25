# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay, transformer_policy
from open_seq2seq.optimizers.novograd import NovoGrad

residual_dense = False  # Enable or disable Dense Residual

base_model = Speech2Text

train_cache_features = True
eval_cache_features = True
cache_format = 'hdf5'
cache_regenerate = False

base_params = {
    "random_seed": 0,
    "use_horovod": False,
    "num_epochs": 30,

    "num_gpus": 1,
    "batch_size_per_gpu": 64,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 1100,
    "save_checkpoint_steps": 1100,
    "logdir": "Oracle_Teacher",
    "num_checkpoints": 5,

    "optimizer": tf.contrib.opt.LazyAdamOptimizer,
    "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
     },
    "lr_policy": transformer_policy,
    "lr_policy_params": {
    "learning_rate": 1.5,
    "warmup_steps": 4000,
    "d_model": 512,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation": [1]
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation": [1],
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [13], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation": [1],
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [17], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation": [1],
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [21], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation": [1],
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [25], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation": [1],
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation": [2]
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation": [1]
            }
        ],

        "dropout_keep_prob": 1.0,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": tf.nn.relu,
        "data_format": "channels_last",
        "use_conv_mask": True,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,
        "infer_logits_to_pickle": False,
    },
    "loss": CTCLoss,
    "loss_params": {},

    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "norm_per_feature": True,
        "pad_to": 16,
        "dither": 1e-5,
        "backend": "librosa",
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "cache_features": train_cache_features,
        "cache_format": cache_format,
        "cache_regenerate": cache_regenerate,
        "dataset_files": [
            "data/librispeech/librivox-train-clean-100.csv",
            "data/librispeech/librivox-train-clean-360.csv",
            "data/librispeech/librivox-train-other-500.csv"
        ],
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "cache_features": eval_cache_features,
        "cache_format": cache_format,
        "cache_regenerate": cache_regenerate,
        "dataset_files": [
            "data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "cache_features": train_cache_features,
        "cache_format": cache_format,
        "cache_regenerate": cache_regenerate,
        "dataset_files": [
            "data/librispeech/librivox-train-clean-100.csv",
            "data/librispeech/librivox-train-clean-360.csv",
            "data/librispeech/librivox-train-other-500.csv"
        ],
        "shuffle": False,
    },
} 
