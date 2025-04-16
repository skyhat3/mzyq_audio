import os
import time
import torch

import datetime
from torch.utils.data import DataLoader,random_split,Dataset
from models.model import PANNS_CNN10,PANNS_CNN14,PANNS_CNN6
from utils import CustomDataset,NumpyDataset


class Trainer:
    def __init__(self,dir,epochs=100,lr=1e-4,bs=4,filename_or_dataset='<CSV_OR_NPYLIST>',num_class=78,input_size=2241,in_channels=1):
        self.epochs=epochs
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(filename_or_dataset,str):
            self.dataset=CustomDataset(filename_or_dataset,dir=dir)
        elif isinstance(filename_or_dataset,list):
            self.dataset=NumpyDataset(*filename_or_dataset)

        val_size = int(len(self.dataset) * 0.2)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        print(len(self.train_dataset),len(self.val_dataset))
        self.train_loader = DataLoader(self.train_dataset, batch_size=bs, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=bs, shuffle=False)
        self.model=PANNS_CNN6(input_size=input_size,num_class=num_class,in_channels=in_channels).to(self.device)
        self.optimizer=torch.optim.AdamW(params=self.model.parameters(),lr=lr,weight_decay=1e-4)
        self.loss_fn=torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=15,gamma=0.1)

    def train(self):
        sum_loss=0.
        total=0
        correct=0
        self.model.train()
        for audios,labels in self.train_loader:

            #print(audios.shape) # torch.Size([16, 32, 2241])
            
            audios=audios.to(self.device)
            labels=labels.to(self.device)
            outputs=self.model(audios)
            loss=self.loss_fn(outputs,labels)
            sum_loss+=loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        avg_loss=sum_loss/len(self.train_loader)
        accuracy = 100 * correct / total
        print('avg loss:', avg_loss)
        print('accuracy:', accuracy, '%')
    
    def run(self):
        start_time = time.time()  # 记录程序开始时间
        for i in range(self.epochs):
            print('epoch:',i+1)
            self.train()
            self.validate()
        end_time = time.time()  # 记录程序结束时间
        elapsed_time = end_time - start_time  # 计算程序运行时间
        print(f"程序运行时间: {elapsed_time:.2f} 秒")
        now=datetime.datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M")
        model_filename = f"model_{date_time_str}.pth"
        torch.save(self.model.state_dict(), os.path.join('<PATH_TO_WEIGHTS_DIR>',model_filename))
        print(f"Model saved as {model_filename}")
    
    def validate(self):
        sum_loss = 0.
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for audios, labels in self.val_loader:
                audios = audios.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(audios)
                loss = self.loss_fn(outputs, labels)
                sum_loss += loss.item()

                # 计算预测的类别
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = sum_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        print('Val avg loss:', avg_loss)
        print('Val accuracy:', accuracy, '%')


if __name__=='__main__':
    csv_file='<CSV>'
    dir=r'<PATH_TO_SEG_AUDIO_DIR>'
    # npy_files=['audio_data.npy','labels.npy']
    trainer=Trainer(epochs=50,filename_or_dataset=csv_file,dir=dir,num_class=78,bs=16,lr=1e-3)
    trainer.run()