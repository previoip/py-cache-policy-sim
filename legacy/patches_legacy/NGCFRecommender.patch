82c82
<         super(NGCF, self).__init__(config)
---
>         self.config = config
84,88c84
<         self.epochs = config['epochs']
<         self.lr = config['lr']
<         self.topk = config['topk']
<         self.user_num = config['user_num']
<         self.item_num = config['item_num']
---
>         super(NGCF, self).__init__(self.config)
90,91c86,95
<         # get this matrix from utils.get_inter_matrix and add it in config
<         self.interaction_matrix = config['inter_matrix']
---
>         self.epochs = self.config['epochs']
>         self.lr = self.config['lr']
>         self.topk = self.config['topk']
>         self.user_num = self.config['user_num']
>         self.item_num = self.config['item_num']
> 
>         # get this matrix from utils.get_inter_matrix and add it in self.config
>         # self.interaction_matrix = self.config['inter_matrix']
>         self._get_interaction_matrix = lambda: self.config['inter_matrix']
>         self.interaction_matrix = None
93,94c97,98
<         self.embedding_size = config['factors']
<         self.hidden_size_list = config["hidden_size_list"] if config['hidden_size_list'] is not None else [64, 64, 64]
---
>         self.embedding_size = self.config['factors']
>         self.hidden_size_list = self.config["hidden_size_list"] if self.config['hidden_size_list'] is not None else [64, 64, 64]
97,100c101,104
<         self.node_dropout = config['node_dropout']
<         self.message_dropout = config['mess_dropout']
<         self.reg_1 = config['reg_1']
<         self.reg_2 = config['reg_2']
---
>         self.node_dropout = self.config['node_dropout']
>         self.message_dropout = self.config['mess_dropout']
>         self.reg_1 = self.config['reg_1']
>         self.reg_2 = self.config['reg_2']
113,116c117,120
<         self.loss_type = config['loss_type']
<         self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
<         self.initializer = config['init_method'] if config['init_method'] != 'default' else 'xavier_normal'
<         self.early_stop = config['early_stop']
---
>         self.loss_type = self.config['loss_type']
>         self.optimizer = self.config['optimizer'] if self.config['optimizer'] != 'default' else 'adam'
>         self.initializer = self.config['init_method'] if self.config['init_method'] != 'default' else 'xavier_normal'
>         self.early_stop = self.config['early_stop']
121c125,127
<         self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
---
>         # self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
>         self._gen_norm_adj_matrix = lambda: self.get_norm_adj_mat().to(self.device)
>         self.norm_adj_matrix = None
124a131,133
>         if self.interaction_matrix is None:
>             self.interaction_matrix = self._get_interaction_matrix()
>        
158a168,169
>         if self.norm_adj_matrix is None:
>             self.norm_adj_matrix = self._gen_norm_adj_matrix()
