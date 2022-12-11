import ml_collections

def get_rna5_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.sde_type = "simple"
  config.data_type = "train"
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.train_bs_x_dsm = 1000
  config.train_bs_t_dsm = 100
  config.train_bs_x = 1000
  config.train_bs_t = 100
  config.num_stage = 80
  config.num_epoch = 10
  config.t0 = 0.0
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'

  config.problem_name = 'rna-5'
  config.num_itr = 100
  config.eval_itr = 200
  config.forward_net = 'OTFlow'
  config.backward_net = 'OTFlow'

  config.num_itr_dsm = 10000
  config.DSM_warmup = False

  # sampling
  config.samp_bs = 1000
  config.sigma_min = 0.01
  config.sigma_max = 5

  # optimization
  #   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'Adam'
  config.lr = 1e-3
  config.lr_gamma = 0.9

  model_configs=None
  return config, model_configs

