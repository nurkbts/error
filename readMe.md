self_scores: torch.Size([64, 8])
attention : torch.Size([8, 64])
attention_heads: torch.Size([64, 64])



RuntimeError                              Traceback (most recent call last)
<ipython-input-1-0452659bf233> in <module>()
     53       for idx_batch, (x, y) in enumerate(train_dl):
     54         optimizer.zero_grad()
---> 55         netout = net(x.to(device))
     56         loss = loss_function(netout, y)
     57         loss.backward()

5 frames
/content/multiHeadAttention.py in forward(self, query, key, value, mask)
     94         if self._attention_size is not None:
     95             attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
---> 96             self._scores = self._scores.masked_fill(attention_mask, float('-inf'))
     97 
     98         # Compute future mask

RuntimeError: The size of tensor a (64) must match the size of tensor b (16) at non-singleton dimension 1
