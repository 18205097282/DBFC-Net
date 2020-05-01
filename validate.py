import sys
import torch
from torch.autograd import Variable
import numpy as np
import pickle
out_sum={}
with open('nums' + '.pkl', 'rb') as f:
    dict_a=pickle.load(f)
with open('label' + '.pkl', 'rb') as f:
    dict_label=pickle.load(f)
for i in dict_a.keys():
    b=np.zeros(200)
    out_sum[i]=b


def validate(loader, model, args, flag):
    model.eval()
    total_output = []
    total_label = []
    start_model = True
    for i, (input, target,name) in enumerate(loader):
        v_id = name[0].split('/')[-1].split(' ')[0].split('.')[0][:-6]
        with torch.no_grad():
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
        target = target.cuda().cpu().detach().numpy()
        # output= model.forward_share(input_var)[0][0].cpu().detach().numpy()
        output = model.forward(input_var)[0][0].cpu().detach().numpy()
        out_sum[v_id]+=output
    count_T=0
    for i in dict_a.keys():
        out_sum[i]/=dict_a[i]
        if np.argmax(out_sum[i])==dict_label[i]:
            count_T+=1
    print(count_T)
    print(count_T/len(dict_a))
    # for i in label_dict.keys():
    #     output=torch.tensor([output_dict[i]])
    #     output = F.softmax(output, dim=1).detach().numpy()
    #     num += output.shape[0]
    #     if(count==0):
    #         f[count*size:num,:] = output
    #     else:
    #         f[count*size:(count+1)*size,:] = output
    #     count+=1
    #
    # np.savetxt('features_te.txt', f)
    # with open('output_v' + '.pkl', 'wb') as f:
    #     pickle.dump(out_sum, f, pickle.HIGHEST_PROTOCOL)
    # with open('label_v' + '.pkl', 'wb') as f1:
    #     pickle.dump(dict_label, f1, pickle.HIGHEST_PROTOCOL)