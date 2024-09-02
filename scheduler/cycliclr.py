import torch

def Scheduler(optimizer, base_lr, lr, cycle_step, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=lr, 
					     						step_size_up=cycle_step // 2, mode='triangular2',
												cycle_momentum=False)

	lr_step = 'iteration'

	print('Initialised Cyclic LR scheduler')

	return sche_fn, lr_step