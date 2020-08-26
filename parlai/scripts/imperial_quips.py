from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging


from parlai.core.build_data import modelzoo_path
from parlai.core.params import get_model_name

import random

class ImperialQuipWorld():

    human_agent = None
    models = None
    models_labels = None
    
    def __init__(self, human_agent, models, models_labels):
        self.human_agent = human_agent
        self.models = models
        self.models_labels = models_labels


    def parley(self):

        human_input = self.human_agent.act()
        suggestions = []
                    
        for model in self.models:
            model.observe(human_input)
            suggestion = model.act()
            suggestions.append(suggestion['text'])

        print("ImperialQuips suggestions:")
        for i in range(len(self.models)):
            print(self.models_labels[i] + ": " +  suggestions[i])
            





# python parlai/scripts/interactive.py -mf zoo:pretrained_transformers/model_poly/model -t convai2
#opt_convai = {'init_opt': None, 'task': 'convai2', 'download_path': '/home/tatiana/ParlAI/downloads', 'loglevel': 'success', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'datapath': '/home/tatiana/ParlAI/data', 'model': None, 'model_file': '/home/tatiana/ParlAI/data/models/pretrained_transformers/model_poly/model', 'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': 1024, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'candidates': 'inline', 'eval_candidates': 'inline', 'interactive_candidates': 'fixed', 'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 'fixed_candidate_vecs': 'reuse', 'encode_candidate_vecs': True, 'encode_candidate_vecs_batchsize': 256, 'train_predict': False, 'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'inference': 'max', 'topk': 5, 'return_cand_scores': False, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False, 'share_encoders': True, 'share_word_embeddings': True, 'learn_embeddings': True, 'data_parallel': False, 'reduction_type': 'mean', 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'polyencoder_type': 'codes', 'poly_n_codes': 64, 'poly_attention_type': 'basic', 'poly_attention_num_heads': 4, 'codes_attention_type': 'basic', 'codes_attention_num_heads': 4, 'display_partner_persona': True, 'parlai_home': '/home/tatiana/ParlAI', 'override': {'model_file': '/home/tatiana/ParlAI/data/models/pretrained_transformers/model_poly/model', 'task': 'convai2'}, 'starttime': 'Aug24_12-51'}
opt_convai = {'activation': 'gelu',
'adafactor_eps': '(1e-30, 0.001)',
'adam_eps': 1e-08,
'add_p1_after_newln': False,
'aggregate_micro': False,
'attention_dropout': 0.0,
'batch_length_range': 5,
'batch_sort_cache_type': 'pop',
'batch_sort_field': 'ext',
'batchsize': 16,
'beam_block_full_context': False,
'beam_block_list_filename': None,
'beam_block_ngram': -1,
'beam_context_block_ngram': -1,
'beam_delay': 30,
'beam_length_penalty': 0.65,
'beam_min_length': 1,
'beam_size': 1,
'betas': [0.9, 0.999],
'bpe_add_prefix_space': None,
'bpe_debug': False,
'bpe_merge': None,
'bpe_vocab': None,
'compute_tokenized_bleu': False,
'datapath': '/home/tatiana/.local/lib/python3.8/site-packages/data',
'datatype': 'train',
'delimiter': '\n',
'dict_class': 'parlai.core.dict:DictionaryAgent',
'dict_endtoken':' __end__',
'dict_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/convai2_ft/model.dict',
'dict_include_test': False,
'dict_include_valid': False,
'dict_initpath': None,
'dict_language': 'english',
'dict_loaded': True,
'dict_lower': True,
'dict_max_ngram_size': -1,
'dict_maxexs': -1,
'dict_maxtokens': -1,
'dict_minfreq': 0,
'dict_nulltoken': '__null__',
'dict_starttoken': '__start__',
'dict_textfields': 'text,labels',
'dict_tokenizer': 'bpe',
'dict_unktoken': '__unk__',
'display_examples': False,
'display_ignore_fields': 'label_candidates,text_candidates',
'display_partner_persona': True,
'display_prettify': False,
'download_path': '/home/tatiana/.local/lib/python3.8/site-packages/downloads',
'dropout': 0.1,
'dynamic_batching': None,
'embedding_projection': 'random',
'embedding_size': 512,
'embedding_type': 'random',
'embeddings_scale': True,
'eval_batchsize': None,
'evaltask': None,
'ffn_size': 2048,
'force_fp16_tokens': False,
'fp16': False,
'fp16_impl': 'apex',
'gpu': -1,
'gradient_clip': 0.1,
'hf_skip_special_tokens': True,
'hide_labels': False,
'history_add_global_end_token': None,
'history_reversed': False,
'history_size': -1,
'image_cropsize': 224,
'image_encoder_num_layers': 1,
'image_features_dim': 2048,
'image_fusion_type': 'late',
'image_mode': None,
'image_size': 256,
'include_image_token': True,
'inference': 'greedy',
'init_model': 'data/models/dodecadialogue/base_model/model',
'init_opt': None,
'interactive_mode': True,
'interactive_task': True,
'invsqrt_lr_decay_gamma': -1,
'label_truncate': 128,
'learn_positional_embeddings': True,
'learningrate': 1e-08,
'load_from_checkpoint': True,
'local_human_candidates_file': None,
'log_every_n_secs': 10.0,
'log_keep_fields': 'all',
'loglevel': 'info',
'lr_scheduler': 'reduceonplateau',
'lr_scheduler_decay': 0.5,
'lr_scheduler_patience': 3,
'max_lr_steps': -1,
'max_train_time': 84600.0,
'metrics': 'default',
'model': 'image_seq2seq',
'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/convai2_ft/model',
'model_parallel': False,
'momentum': 0,
'ultitask_weights': [1],
'n_decoder_layers': -1,
'n_encoder_layers': -1,
'n_heads': 16,
'n_image_channels': 1,
'n_image_tokens': 1,
'n_layers': 8,
'n_positions': 512,
'n_segments': 0,
'nesterov': True,
'no_cuda': False,
'num_epochs': -1,
'numthreads': 1,
'numworkers': 4,
'nus': [0.7],
'optimizer': 'adamax',
'outfile': None,
'output_scaling': 1.0,
'override': {'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/convai2_ft/model', 'task': 'convai2'},
'parlai_home': '/checkpoint/kshuster/projects/parlall/parlall_MT_plus_FT/parlall_MT_plus_FT_sweep1_Tue_Oct_29/ParlAI',
'person_tokens': False,
'pytorch_context_length': -1,
'pytorch_datapath': None,
'pytorch_include_labels': True,
'pytorch_preprocess': False,
'pytorch_teacher_batch_sort': False,
'pytorch_teacher_dataset': None,
'pytorch_teacher_task': None,
'rank_candidates': False,
'relu_dropout': 0.0,
'save_after_valid': True,
'save_every_n_secs': -1,
'save_format': 'conversations',
'share_word_embeddings': True,
'short_final_eval': False,
'show_advanced_args': False,
'shuffle': False,
'single_turn': False,
'skip_generation': True,
'special_tok_lst': None,
'split_lines': False,
'starttime': 'Oct29_07-56',
'task': 'convai2',
'temperature': 1.0,
'tensorboard_log': False,
'text_truncate': 512,
'topk': 10,
'topp': 0.9,
'truncate': -1,
'update_freq': 1,
'use_reply': 'label',
'validation_cutoff': 1.0,
'validation_every_n_epochs': -1,
'validation_every_n_secs': 3600.0,
'validation_max_exs': -1,
'validation_metric': 'ppl',
'validation_metric_mode': min,
'validation_patience': 10,
'validation_share_agent': False,
'variant': 'xlm',
'warmup_rate': 0.0001,
'warmup_updates': 2000,
'weight_decay': None
}

#python parlai interactive -mf zoo:dodecadialogue/empathetic_dialogues_ft/model
#opt_ed = {'init_opt': None, 'task': 'interactive', 'download_path': '/home/tatiana/ParlAI/downloads', 'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'datapath': '/home/tatiana/ParlAI/data', 'model': None, 'model_file': '/home/tatiana/ParlAI/data/models/dodecadialogue/empathetic_dialogues_ft/model', 'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False, 'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'image_features_dim': 2048, 'image_encoder_num_layers': 1, 'n_image_tokens': 1, 'n_image_channels': 1, 'include_image_token': True, 'image_fusion_type': 'late', 'parlai_home': '/home/tatiana/ParlAI', 'override': {'model_file': '/home/tatiana/ParlAI/data/models/dodecadialogue/empathetic_dialogues_ft/model'}, 'starttime': 'Aug24_14-42'}
opt_ed = {'activation': 'gelu',
'adafactor_eps': '(1e-30, 0.001)',
'adam_eps': 1e-08,
'add_p1_after_newln': False,
'aggregate_micro': False,
'attention_dropout': 0.0,
'batch_length_range': 5,
'batch_sort_cache_type': 'pop',
'batch_sort_field': 'ext',
'batchsize': 16,
'beam_block_full_context': False,
'beam_block_list_filename': None,
'beam_block_ngram': -1,
'beam_context_block_ngram': -1,
'beam_delay': 30,
'beam_length_penalty': 0.65,
'beam_min_length': 1,
'beam_size': 1,
'betas': [0.9, 0.999],
'bpe_add_prefix_space': None,
'bpe_debug': False,
'bpe_merge': None,
'bpe_vocab': None,
'compute_tokenized_bleu': False,
'datapath': '/home/tatiana/.local/lib/python3.8/site-packages/data',
'datatype': 'train',
'delimiter': '\n',
'dict_class': 'parlai.core.dict:DictionaryAgent',
'dict_endtoken':' __end__',
'dict_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/empathetic_dialogues_ft/model.dict',
'dict_include_test': False,
'dict_include_valid': False,
'dict_initpath': None,
'dict_language': 'english',
'dict_loaded': True,
'dict_lower': True,
'dict_max_ngram_size': -1,
'dict_maxexs': -1,
'dict_maxtokens': -1,
'dict_minfreq': 0,
'dict_nulltoken': '__null__',
'dict_starttoken': '__start__',
'dict_textfields': 'text,labels',
'dict_tokenizer': 'bpe',
'dict_unktoken': '__unk__',
'display_examples': False,
'display_ignore_fields': 'label_candidates,text_candidates',
'display_partner_persona': True,
'display_prettify': False,
'download_path': '/home/tatiana/.local/lib/python3.8/site-packages/downloads',
'dropout': 0.1,
'dynamic_batching': None,
'embedding_projection': 'random',
'embedding_size': 512,
'embedding_type': 'random',
'embeddings_scale': True,
'eval_batchsize': None,
'evaltask': None,
'ffn_size': 2048,
'force_fp16_tokens': False,
'fp16': False,
'fp16_impl': 'apex',
'gpu': -1,
'gradient_clip': 0.1,
'hf_skip_special_tokens': True,
'hide_labels': False,
'history_add_global_end_token': None,
'history_reversed': False,
'history_size': -1,
'image_cropsize': 224,
'image_encoder_num_layers': 1,
'image_features_dim': 2048,
'image_fusion_type': 'late',
'image_mode': None,
'image_size': 256,
'include_image_token': True,
'inference': 'greedy',
'init_model': 'data/models/dodecadialogue/base_model/model',
'init_opt': None,
'interactive_mode': True,
'interactive_task': True,
'invsqrt_lr_decay_gamma': -1,
'label_truncate': 128,
'learn_positional_embeddings': True,
'learningrate': 1e-08,
'load_from_checkpoint': True,
'local_human_candidates_file': None,
'log_every_n_secs': 10.0,
'log_keep_fields': 'all',
'loglevel': 'info',
'lr_scheduler': 'reduceonplateau',
'lr_scheduler_decay': 0.5,
'lr_scheduler_patience': 3,
'max_lr_steps': -1,
'max_train_time': 84600.0,
'metrics': 'default',
'model': 'image_seq2seq',
'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/empathetic_dialogues_ft/model',
'model_parallel': False,
'momentum': 0,
'ultitask_weights': [1],
'n_decoder_layers': -1,
'n_encoder_layers': -1,
'n_heads': 16,
'n_image_channels': 1,
'n_image_tokens': 1,
'n_layers': 8,
'n_positions': 512,
'n_segments': 0,
'nesterov': True,
'no_cuda': False,
'num_epochs': -1,
'numthreads': 1,
'numworkers': 4,
'nus': [0.7],
'optimizer': 'adamax',
'outfile': None,
'output_scaling': 1.0,
'override': {'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/dodecadialogue/empathetic_dialogues_ft/model'},
'parlai_home': '/checkpoint/kshuster/projects/parlall/parlall_MT_plus_FT/parlall_MT_plus_FT_sweep1_Tue_Oct_29/ParlAI',
'person_tokens': False,
'pytorch_context_length': -1,
'pytorch_datapath': None,
'pytorch_include_labels': True,
'pytorch_preprocess': False,
'pytorch_teacher_batch_sort': False,
'pytorch_teacher_dataset': None,
'pytorch_teacher_task': None,
'rank_candidates': False,
'relu_dropout': 0.0,
'save_after_valid': True,
'save_every_n_secs': -1,
'save_format': 'conversations',
'share_word_embeddings': True,
'short_final_eval': False,
'show_advanced_args': False,
'shuffle': False,
'single_turn': False,
'skip_generation': True,
'special_tok_lst': None,
'split_lines': False,
'starttime': 'Oct29_07-56',
'task': 'convai2',
'temperature': 1.0,
'tensorboard_log': False,
'text_truncate': 512,
'topk': 10,
'topp': 0.9,
'truncate': -1,
'update_freq': 1,
'use_reply': 'label',
'validation_cutoff': 1.0,
'validation_every_n_epochs': -1,
'validation_every_n_secs': 3600.0,
'validation_max_exs': -1,
'validation_metric': 'ppl',
'validation_metric_mode': min,
'validation_patience': 10,
'validation_share_agent': False,
'variant': 'xlm',
'warmup_rate': 0.0001,
'warmup_updates': 2000,
'weight_decay': None
}

# python parlai interactive -mf zoo:blended_skill_talk/bst_single_task/model -t blended_skill_talk
#opt_bst = {'init_opt': None, 'task': 'blended_skill_talk', 'download_path': '/home/tatiana/ParlAI/downloads', 'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'datapath': '/home/tatiana/ParlAI/data', 'model': None, 'model_file': '/home/tatiana/ParlAI/data/models/blended_skill_talk/bst_single_task/model', 'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': 1024, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'candidates': 'inline', 'eval_candidates': 'inline', 'interactive_candidates': 'fixed', 'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 'fixed_candidate_vecs': 'reuse', 'encode_candidate_vecs': True, 'encode_candidate_vecs_batchsize': 256, 'train_predict': False, 'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'inference': 'max', 'topk': 5, 'return_cand_scores': False, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False, 'share_encoders': True, 'share_word_embeddings': True, 'learn_embeddings': True, 'data_parallel': False, 'reduction_type': 'mean', 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'polyencoder_type': 'codes', 'poly_n_codes': 64, 'poly_attention_type': 'basic', 'poly_attention_num_heads': 4, 'codes_attention_type': 'basic', 'codes_attention_num_heads': 4, 'display_partner_persona': True, 'include_personas': True, 'include_initial_utterances': False, 'safe_personas_only': True, 'parlai_home': '/home/tatiana/ParlAI', 'override': {'model_file': '/home/tatiana/ParlAI/data/models/blended_skill_talk/bst_single_task/model', 'task': 'blended_skill_talk'}, 'starttime': 'Aug24_13-01'}
opt_bst = {'init_opt': None, 'task': 'blended_skill_talk', 'download_path': '/home/tatiana/.local/lib/python3.8/site-packages/downloads', 'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'datapath': '/home/tatiana/.local/lib/python3.8/site-packages/data', 'model': None, 'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/blender/blender_90M/model', 'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'safety': 'all', 'local_human_candidates_file': None, 'single_turn': False, 'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False, 'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'display_partner_persona': True, 'include_personas': True, 'include_initial_utterances': False, 'safe_personas_only': True, 'parlai_home': '/home/tatiana/.local/lib/python3.8/site-packages', 'override': {'model_file': '/home/tatiana/.local/lib/python3.8/site-packages/data/models/blender/blender_90M/model', 'task': 'blended_skill_talk'}, 'starttime': 'Aug26_02-41'}

#python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent




def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'Interactive chat with a model on the command line'
        )
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    WorldLogger.add_cmdline_args(parser)
    return parser


def interactive(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task

    human_agent = LocalHumanAgent(opt_convai)
        
    convai_agent = create_agent(opt_convai, requireModelExists=True)
    ed_agent = create_agent(opt_ed, requireModelExists=True)
    bst_agent = create_agent(opt_bst, requireModelExists=True)

    models = [convai_agent,ed_agent,bst_agent]
    labels = ['CONVAI2','EMPATHETIC_DIALOGUES','BLENDED_SKILL_TALK']
    
    imperial_quips = ImperialQuipWorld(human_agent, models, labels)

    keep_suggesting = True
    while(keep_suggesting):
        imperial_quips.parley()
        user_input = input("")
        if user_input == "EXIT":
            keep_suggesting = False



@register_script('interactive', aliases=['i'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()
        

    def run(self):
        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()



