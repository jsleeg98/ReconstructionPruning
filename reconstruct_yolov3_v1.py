'''
make_cfg.py와 make_pth.py를 합쳐서 한번의 실행으로 .cfg와 .pth가 생성될 수 있는 파일
'''

from pytorchyolo.utils.parse_config import parse_data_config
import torch
import numpy as np
from collections import OrderedDict
from pytorchyolo.models import *

# cfg 생성 시 활용
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n') # 줄바꿈 단위로 나눈다.
    lines = [x for x in lines if x and not x.startswith('#')]  # 라인에 글이 있고 #으로 시작하지 않으면 line으로 list화한다.
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces #각 줄에서 앞뒤 공백은 모두 제거한다.
    module_defs = []
    for line in lines:
        
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
            #수정    
            if module_defs[-1]['type'] == 'predict':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    
    return module_defs

def main():
    # 실행 설정 값
    #-----------------------------------------------------------------------------#

    # FPGM 압축된 pth
    # pretrained = './FPGM/weight/weight_decay/0.001/yolov3_just_FPGM_0.9.pth'
    pretrained = './AISOC/train_result/original_5/compress/retrain_0.9/weight/yolov3_retrain_BEST.pth'
    # 생성된 실제 압축된 pth
    # real_compressed_pth = './FPGM/weight/weight_decay/0.001/yolov3_real_compressed_0.9.pth'
    real_compressed_pth = './AISOC/train_result/original_5/pruning_weight/0.9.pth'

    # 원본 cfg
    cfg_origin = './AISOC/config/cfg/weight_decay/0.000001/yolov3_predict.cfg'
    # 생성될 cfg
    cfg_compressed = './AISOC/train_result/original_5/cfg/0.9.cfg'

    #------------------------------------------------------------------------------#



    # non_zero_state_dict 얻기
    #-------------------------------------------------------------------------------#
    device = 'cpu'

    checkpoint = torch.load(pretrained, map_location = device)
            # model.load_state_dict(checkpoint['state_dict'])

    state_dict = OrderedDict()
    non_zero_index_dict = OrderedDict()

    cnt = 0
    non_zero_index = torch.tensor([0, 1, 2])
    for name, param in checkpoint.items():
        if 'running_mean' in name or 'running_var' in name or'num_batches_tracked' in name:
            continue

        print(name + ' -> ' + str(cnt))
        print(param.size())
        
        li_name = name.split('.')
      
        
        if not ('pred' in name):
            if 'conv' in li_name[-2]:
                # conv 인 경우
                li = list(param.size())
                a = []
                for i in range(li[0]):
                    a.append(torch.index_select(param[i], 0, non_zero_index))
                tensor = torch.stack(a, 0)

                viewed_param = tensor.view(tensor.size()[0], -1)
                norm = torch.norm(viewed_param, 2, 1) # norm 변환
                non_zero_index = torch.nonzero(norm)
                non_zero_index = non_zero_index.squeeze()
                layer_state_dict = torch.index_select(tensor, 0, non_zero_index)
                state_dict['conv_{}'.format(cnt)] = layer_state_dict
                non_zero_index_dict['non_zero_index_{}'.format(cnt)] = non_zero_index
            elif 'batch_norm' in li_name[-2] and li_name[-1] == 'bias':
                cnt += 1
    

        elif 'pred' in name:
            if li_name[-1] == 'weight' and not ('batch_norm' in li_name[-2]):
                # pred의 conv인 경우
                li = list(param.size())
                a = []
                for i in range(li[0]):
                    a.append(torch.index_select(param[i], 0, non_zero_index))
                tensor = torch.stack(a, 0)
                non_zero_index = torch.tensor(np.arange(li[0]))
                state_dict['pred_{}'.format(cnt)] = tensor
                non_zero_index_dict['non_zero_index_{}'.format(cnt)] = non_zero_index
            elif 'batch_norm' in li_name[-2] and li_name[-1] == 'bias':
                cnt += 1
    #-------------------------------------------------------------------------------------------------#

    #cfg 생성
    #---------------------------------------------------------------------------------------------#
    
    # 생성될 파일
    write_file = open(cfg_compressed, 'w')

    # 기존 cfg에서 원본 모델 구조 가져오기
    module_defs = parse_model_config(cfg_origin)

    # FPGM 적용된 모델에서 non_zero_index_dict 가져오기
    # model_dict = torch.load('./FPGM/weights/non_zero_index.pth')
    # non_zero_index_dict = model_dict['non_zero_index_dict']

    

    # cnt는 convolutional이나 predict일 때만 +1 된다.
    cnt = 0
    for i, module in enumerate(module_defs):
        if module['type'] == 'net':
            continue
        elif module['type'] == 'convolutional':
            # print('cnt = ' + str(cnt))
            filter = len(non_zero_index_dict['non_zero_index_{}'.format(cnt)])
            module['filters'] = filter
            non_zero_index = ','.join(list(map(str, non_zero_index_dict['non_zero_index_{}'.format(cnt)].tolist())))
            # print(non_zero_index)
            module['non_zero_index'] = non_zero_index
            cnt += 1
            
        elif module['type'] == 'shortcut':
            # shortcut이 연속으로 있는 경우 대비
            shortcut_layer = []
            num = 3
            while module_defs[i - num]['type'] == 'shortcut':
                shortcut_layer.append(1 + num)
                num += 3
            shortcut_layer.append(num)
            
            or_index = set(list(map(int, list(module_defs[i - 1]['non_zero_index'].split(',')))))
            for j in shortcut_layer:
                front_index = set(list(map(int, list(module_defs[i - j]['non_zero_index'].split(',')))))
                or_index = or_index | front_index
            or_index = list(or_index)
           
            filter = len(or_index)
            non_zero_index = ','.join(list(map(str,or_index)))
            
            module_defs[i - 1]['non_zero_index'] = non_zero_index
            module_defs[i - 1]['filters'] = filter
            for j in shortcut_layer:
                module_defs[i - j]['non_zero_index'] = non_zero_index
                module_defs[i - j]['filters'] = filter
            
            # import pdb; pdb.set_trace()
        elif module['type'] == 'predict':
            # FPGM 적용하지 않아서 filter를 그대로 둔다.
            # input_channel을 위해 index만 가지고 있는다.
            # print('cnt = ' + str(cnt))
            non_zero_index = ','.join(list(map(str, non_zero_index_dict['non_zero_index_{}'.format(cnt)].tolist())))
            module['non_zero_index'] = non_zero_index
            cnt += 1
        
        elif module['type'] == 'route':
            # 이 경우에는 [-4], [-1, 61], [-4], [-1, 36]의 경우가 있다
            # 개수가 1개와 두개로 나누어 우선 처리

            # -4인 경우
            if len(module['layers'].split(',')) == 1:
                module['non_zero_index'] = module_defs[i - 4]['non_zero_index']

            # -1, 61 또는 -1, 36인 경우
            elif len(module['layers'].split(',')) == 2:
                a, b = map(int, module['layers'].split(','))
                if b == 61:
                    non_zero_a = list(map(int, module_defs[i + a - 1]['non_zero_index'].split(',')))
                    non_zero_b = list(map(int, module_defs[b]['non_zero_index'].split(',')))

                    for i in range(len(non_zero_b)): # 원래 256 + 512 concat
                        non_zero_b[i] += 256
                    
                    non_zero_index = non_zero_a + non_zero_b
                    module['non_zero_index'] = ','.join(map(str, non_zero_index))
                    
                elif b == 36:
                    non_zero_a = list(map(int, module_defs[i + a - 1]['non_zero_index'].split(',')))
                    non_zero_b = list(map(int, module_defs[b]['non_zero_index'].split(',')))

                    for i in range(len(non_zero_b)): # 원래 128 + 256 concat
                        non_zero_b[i] += 128
                    
                    non_zero_index = non_zero_a + non_zero_b
                    module['non_zero_index'] = ','.join(map(str, non_zero_index))


                
        
    line = ''
    for module in module_defs:
        if module['type'] == 'net':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'convolutional':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n'
        elif module['type'] == 'shortcut':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n'
        elif module['type'] == 'predict':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n' 
                #str은 yolo 직전 75 filter에 bn이 없어서 정수 0으로 되어있기 때문
        elif module['type'] == 'yolo':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'route':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'upsample':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        
    write_file.write(line)
    # print(line)

    write_file.close()
    
    #----------------------------------------------------------------------------------#

    # .pth 생성------------------------------------------------------------------------------#

    model_origin = Darknet(cfg_origin)
    model_origin.load_state_dict(torch.load(pretrained, map_location = device)) # FPGM 압축한 가중치

    model_compressed = Darknet(cfg_compressed)

    module_defs = parse_model_config(cfg_compressed)

    non_zero_index = ['0,1,2']
    for module in module_defs:
        # import pdb; pdb.set_trace()
        if 'non_zero_index' in module:
            non_zero_index.append(module['non_zero_index'])

    # named_parameters() 대신에 state_dict().items()를 사용하는 이유는 running_mean, running_var도 복사해야하기 때문이다.
    non_zero_index_i = 0
    for model_ori, model_comp in zip(model_origin.state_dict().items(), model_compressed.state_dict().items()):
        print(model_ori[0], non_zero_index_i)

        # route가 중간에 4개 끼어 있는데 이때 non_zero_index_i를 보정하기 위해서 +1을 해주어 맞춰준다.
        if non_zero_index_i == 59:
            non_zero_index_i += 1 # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
            # import pdb; pdb.set_trace()
        elif non_zero_index_i == 61:
            non_zero_index_i += 1 # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
            # import pdb; pdb.set_trace()
        elif non_zero_index_i == 69:
            non_zero_index_i += 1 # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
        elif non_zero_index_i == 71:
            non_zero_index_i += 1 # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
        
        if len(model_ori[1].size()) == 4:
            channel_index = torch.tensor(list(map(int,non_zero_index[non_zero_index_i].split(','))))
            filter_index = torch.tensor(list(map(int,non_zero_index[non_zero_index_i+1].split(','))))
            # import pdb; pdb.set_trace()
            temp_weight = torch.index_select(model_ori[1], 1, channel_index)
            temp_weight = torch.index_select(temp_weight, 0, filter_index)
            # import pdb; pdb.set_trace()
            model_comp[1].data.copy_(temp_weight.data) # weight 값 깊은 복사

        elif len(model_ori[1].size()) == 1: # bn_weight와 bn_bias
            filter_index = torch.tensor(list(map(int,non_zero_index[non_zero_index_i+1].split(','))))
            temp_weight = torch.index_select(model_ori[1], 0, filter_index) 

            model_comp[1].data.copy_(temp_weight.data) # weight 값 깊은 복사
            
            # if 'bias' in model_ori[0] and not('pred_81' in model_ori[0]) and not('pred_93' in model_ori[0]) and not('pred_105' in model_ori[0]): # bn_bias인 경우 
            #     non_zero_index_i += 1 # 다음 non_zero_index를 참조하기 위함
            #     # import pdb; pdb.set_trace()
        elif len(model_ori[1].size()) == 0: # num_batches_tracked
            non_zero_index_i += 1

    


    # import pdb; pdb.set_trace()
    # for name, param in model_compressed.named_parameters():
        # print(name)
        # import pdb; pdb.set_trace()

    torch.save(model_compressed.state_dict(), real_compressed_pth)
    print('모델 저장 완료')
    

    
    
  
if __name__ == "__main__":
    main()
