# Colorization
 
## Training process
```bash
python train.py  -b 100  -lr 0.001  -m 1000
```

## Inference process
```bash
python demo.py  -model [MODELPATH]  -img [IMGPATH]
```
where `[MODELPATH]` means the path of model, `[IMGPATH]` means the path of gray image.

For instance,
```bash
python demo.py  -model model_cpu.pth  -img landscape_images/10.jpg
```

The output image will be automatically saved in `output.jpg`.