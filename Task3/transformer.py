import torchvision.transforms as T

class ImgTransformer:
    def __init__(self):
        self.transformer = T.Compose([
            T.Lambda(self.rgb_conv),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def __call__(self, *args, **kwds):
        return self.transformer(*args, **kwds)
    
    def rgb_conv(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img
