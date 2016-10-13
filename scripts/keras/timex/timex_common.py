import nn_models as models

def get_model_for_config(dimension, vocab_size, num_outputs, config, weights = None):
    if not weights is None and not weights.shape[1] == config['embed_dim']:
        print("WARNING: Pre-trained embedding weights have dimensionality %d which is different than config embedding dimensionality %d -- modifying config" % (weights.shape[1], config['embed_dim']) )
        config['embed_dim'] = weights.shape[1]
    
    if config['bilstm']:
        model = models.get_bio_bilstm_model(dimension=dimension, vocab_size=vocab_size, num_outputs=num_outputs, layers=config['layers'], embed_dim=config['embed_dim'], activation=config['activation'], go_backwards=config['backwards'], weights=weights, lr=config['lr'])
    else:
        model = models.get_bio_lstm_model(dimension=dimension, vocab_size=vocab_size, num_outputs=num_outputs, layers=config['layers'], embed_dim=config['embed_dim'], activation=config['activation'], go_backwards=config['backwards'], weights=weights, lr=config['lr'])
    
    return model


