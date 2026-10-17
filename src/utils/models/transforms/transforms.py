from torchvision import transforms

def simple_resize_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
