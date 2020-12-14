local embedding_dim = 300;
local hidden_dim = 250;
local early_stopping = 10;
local batch_size = 10;
local learning_rate = 0.0001;
local max_decoding_steps = 40;
local beam_size = 1;
local num_enc_layers = 1;
local input_dropout = 0.2;
local gpus = 0;

{
  "dataset_reader": {
    "type": "copynet_shared_decoder",
    "target_namespace": "tokens",
    "lazy": true
  },
  "model": {
    "type": "copynet_shared_decoder",
    "source_embedder": {
      "type": "basic",
      "token_embedders": {
	"tokens": {
	  "type": "embedding",
	  "embedding_dim": embedding_dim
	}
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "num_layers": num_enc_layers,
      "batch_first": true,
      "bidirectional": true
    },
    "attention": {
      "type": "dot_product"
    },
    "beam_size": beam_size,
    "max_decoding_steps": max_decoding_steps,
    "dropout": input_dropout,
    "source_namespace": "tokens",
    "target_namespace": "tokens"
  },
  "iterator": {
    "type": "homogeneous_batch",
    "batch_size": batch_size,
    "max_instances_in_memory": 800,
    "partition_key": "target_language"
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": learning_rate
    },
    "patience": early_stopping,
    "num_epochs": 1,
    "num_serialized_models_to_keep": 1,
    "cuda_device": gpus,
    "shuffle": true
  },
  "vocabulary": {
    "directory_path": "vocabularies/webNLG"
  }
}
