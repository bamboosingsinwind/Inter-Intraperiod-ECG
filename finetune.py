import os
import time
from xml.parsers.expat import model
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from dataset import train_set,valid_set,test_set
# from dataset_cinc2017 import train_set,valid_set,test_set
# from dataset_cpsc2021 import train_set,valid_set,test_set
from labelSmoothing import LabelSmoothing
from sklearn.metrics import roc_auc_score,roc_curve,f1_score,confusion_matrix
from  onedResnet_myopt import resnet18to34

def main():
    dir_path = "./Inter-Intraperiod-ECG/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 4  #64#16#32#128#4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #==========dataset==============
    train_loader = torch.utils.data.DataLoader(train_set,#train_dataset
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=8)
    validate_loader = torch.utils.data.DataLoader(valid_set,#validate_dataset
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,#validate_dataset
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=8)

    val_num = len(valid_set)
    test_num = len(test_set)
    #==========model==============
    net = resnet18to34(num_classes=2)
    save_model_path = "./save_model/best.pth"
    save_model = torch.load(save_model_path)
    model_dict = net.state_dict()
    state_dict = {k.replace("encoder.",""):v for k,v in save_model.items() if k.replace("encoder.","") in model_dict.keys()}
 
    state_dict.pop('fc.weight')
    state_dict.pop('fc.bias')
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    
    # for param in net.parameters():
    #     param.requires_grad = False
    # for param in net.fc.parameters():
    #     print("param",param)
    #     param.requires_grad = True
    # net = resnet18to34(num_classes=2)

    net = net.to(device)
   
    #===========loss function & optimizer==============
    loss_function = LabelSmoothing(0.1)
    # loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adagrad(net.parameters(), lr=0.0003)
    optimizer = optim.Adam(params, lr=0.0003)# , lr=0.0003  0.0003
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    epochs = 100
    model_name = "fine_tune"
    save_path = dir_path + "save_models/"+model_name+".pth"
    save_test_path = dir_path + "save_models/"+model_name+"_test.pth"
    logs_path = dir_path + "./logs/log.txt"  
    with open(logs_path,"a+") as f:
        f.write(model_name+str(time.strftime('%Y-%m-%d %H:%M:%S'))+"\n")
    #train==================================
    train_steps = len(train_loader)
    best_auc = 0.0
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            sigs,feat_plus, labels = data
            optimizer.zero_grad()
            logits = net(sigs.to(device),feat_plus.to(device))
            
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     running_loss/len(train_set))
        scheduler.step()
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        target_label = []
        pred_score = []
        pred_label = []
        running_loss = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_sigs, feat_plus,val_labels = val_data
                outputs= net(val_sigs.to(device),feat_plus.to(device))
                pred = nn.Softmax(dim=1)(outputs)[:,1:2]
                pred_score.extend(list(np.array(pred.squeeze(0).cpu())))
                target_label.extend(list(np.array(val_labels.cpu())))

                loss = loss_function(outputs, val_labels.to(device))
                running_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                pred_label.extend(list(np.array(predict_y.cpu())))
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(valid_set), val_accurate))
        cm = confusion_matrix(target_label,pred_label)
        print("confusion matrix \n",cm)
        auc = roc_auc_score(target_label,pred_score)
        print("auc=",auc)
        f1 = f1_score(target_label,pred_label)
        print("f1_score=",f1)
        if ( auc > best_auc):
            best_auc = auc
            torch.save(net.state_dict(), save_path)
            print("update saving the model")
        with open(logs_path,"a+") as f:
            f.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
            f.write("confusion matrix \n"+str(cm))
            f.write("auc="+str(auc)+"\n")
            f.write("f1_score"+str(f1)+"\n")
    print('Finished Training')
    
    # test=======================================
    net.load_state_dict(torch.load(save_path))
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    target_label = []
    pred_score = []
    pred_label = []
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_sigs, feat_plus,test_labels = test_data
            outputs= net(test_sigs.to(device),feat_plus.to(device))
            pred = nn.Softmax(dim=1)(outputs)[:,1:2]
            pred_score.extend(list(np.array(pred.squeeze(0).cpu())))
            target_label.extend(list(np.array(test_labels.cpu())))
            predict_y = torch.max(outputs, dim=1)[1]
            pred_label.extend(list(np.array(predict_y.cpu())))
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    test_accurate = acc / test_num
    print('test accuracy=', test_accurate)
    cm = confusion_matrix(target_label,pred_label)
    print("confusion matrix \n",cm)
    auc = roc_auc_score(target_label,pred_score)
    print("auc=",auc)
    f1 = f1_score(target_label,pred_label)

    with open(logs_path,"a+") as f:
        f.write("test result \n")
        f.write("test accuracy="+str(test_accurate)+"\n")
        f.write("confusion matrix \n"+str(cm))
        f.write("auc="+str(auc)+"\n")
        f.write("f1_score"+str(f1)+"\n")
    #==========roc curve=================
    fpr,tpr,_ = roc_curve(target_label,pred_score,pos_label=1)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area={:.3f})'.format(auc))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
    # plt.savefig(dir_path + "figures/"+model_name+"_roc_curve.jpg")
 
      
if __name__ == '__main__':
    main()
