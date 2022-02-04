from basic_model import Net
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import argparse

MODELPATH = 'model_cpu.pth'
IMGPATH = ''

class UnNormalize:
    #restore from T.Normalize
    def __init__(self,mean=(0.5, 0.5, 0.5),std= (0.5, 0.5, 0.5)):
        self.mean = torch.tensor(mean).view((1,-1,1,1))
        self.std = torch.tensor(std).view((1,-1,1,1))
    def __call__(self, x):
        x = (x * self.std) + self.mean
        return torch.clip(x, 0, None)

def main():
    # load model.
    model = Net()
    model.load_state_dict(torch.load(MODELPATH))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load image.
    img = Image.open(IMGPATH).convert('RGB')
    input_transform = T.Compose([T.ToTensor(),
                                 T.Grayscale(),
                                 T.Normalize((0.5), (0.5))
                                 ])
    input_img = input_transform(img)
    input_img = input_img.unsqueeze(0)

    # predict.
    output_img = model.forward(input_img)
    output_img = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(output_img)
    output_img = output_img.squeeze()
    output_img = T.ToPILImage()(output_img)

    # show & save
    output_img.show()
    output_img.save('output.jpg')

    return

def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', metavar='MODELPATH', help='the model path')
    parser.add_argument('-img', metavar='IMGPATH', help='the path of image')
    args = parser.parse_args()
    global  MODELPATH
    global  IMGPATH
    if args.model: MODELPATH = args.model
    if args.img: IMGPATH = args.img

    return

if __name__ == '__main__':
    ArgParser()
    main()