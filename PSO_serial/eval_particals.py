def eval_particals(self):
    min_lenth = min(self.lenths)
    min_index = self.lenths.index(min_lenth)
    cur_path = self.particals[min_index]
    # Update the current global optimum solution
    if min_lenth < self.global_best_len:
        self.global_best_len = min_lenth
        self.global_best = cur_path
    # Update the current individual optimal solution
    for i, l in enumerate(self.lenths):
        if l < self.local_best_len[i]:
            self.local_best_len[i] = l
            self.local_best[i] = self.particals[i]