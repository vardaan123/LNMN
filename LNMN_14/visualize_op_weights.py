import torch
import torch.nn.functional as F
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Stack-NMN')

	args = parser.parse_args()
	args.ckpt = 'config_1_cloud/ckpt_epoch_xx.model'

	vis_mat_3input = torch.zeros(5*2, 6)
	vis_mat_4input = torch.zeros(4*3, 6)

	vis_mat_3input_ans = torch.zeros(2*2, 6)
	vis_mat_4input_ans = torch.zeros(2*3, 6)

	model = torch.load(args.ckpt, map_location='cpu')

	for i in range(5):
		for j in range(2):
			vis_mat_3input[i*2+j, :] = model['module_list.{}.node_list.{}.op_weights'.format(i,j)]

	for i in range(5,9):
		for j in range(3):
			vis_mat_4input[(i-5)*3+j, :] = model['module_list.{}.node_list.{}.op_weights'.format(i,j)]

	for i in range(9,11):
		for j in range(2):
			vis_mat_3input_ans[(i-9)*2+j, :] = model['module_list.{}.node_list.{}.op_weights'.format(i,j)]

	for i in range(11,13):
		for j in range(3):
			vis_mat_4input_ans[(i-11)*3+j, :] = model['module_list.{}.node_list.{}.op_weights'.format(i,j)]


	# apply softmax
	vis_mat_3input = F.softmax(vis_mat_3input, dim=1)
	vis_mat_4input = F.softmax(vis_mat_4input, dim=1)
	vis_mat_3input_ans = F.softmax(vis_mat_3input_ans, dim=1)
	vis_mat_4input_ans = F.softmax(vis_mat_4input_ans, dim=1)
		
	print(vis_mat_3input)
	print(vis_mat_4input)
	print(vis_mat_3input_ans)
	print(vis_mat_4input_ans)

plt.figure()
plt.matshow(vis_mat_3input.numpy())
plt.axis('off')
plt.colorbar()
plt.savefig('vis_mat_3input.png', transparent=True)
plt.close()

plt.figure()
plt.matshow(vis_mat_4input.numpy())
plt.axis('off')
plt.colorbar()
plt.savefig('vis_mat_4input.png', transparent=True)

plt.figure()
plt.matshow(vis_mat_3input_ans.numpy())
plt.axis('off')
plt.colorbar()
plt.savefig('vis_mat_3input_ans.png', transparent=True)

plt.figure()
plt.matshow(vis_mat_4input_ans.numpy())
plt.axis('off')
plt.colorbar()
plt.savefig('vis_mat_4input_ans.png', transparent=True)
