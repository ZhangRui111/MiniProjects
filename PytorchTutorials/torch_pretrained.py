import torch
import torchvision
import torchsummary


def pretrained_model():
    """
    Download and load the pretrained ResNet-18.
    :return:
    """
    resnet34 = torchvision.models.resnet34(pretrained=True).to(device)
    # Finetune only the top layer of the model.
    for param in resnet34.parameters():
        param.requires_grad = False
    # Replace the top layer for finetuning.
    # in_features: size of each input sample
    resnet34.fc = torch.nn.Linear(resnet34.fc.in_features, 100)
    # Forward pass.
    images = torch.randn(64, 3, 224, 224)
    outputs = resnet34(images)
    print(outputs.size())


def save_load_model():
    resnet34 = torchvision.models.resnet34(pretrained=True).to(device)
    # print(resnet34)
    # torchsummary.summary(resnet34, input_size=(3, 224, 224))

    # Save and load the entire model.
    torch.save(resnet34, "./logs/model/model.ckpt")
    load_model = torch.load("./logs/model/model.ckpt").to(device)
    # print(load_model)
    torchsummary.summary(load_model, input_size=(3, 224, 224))

    # Save and load only the model parameters (recommended).
    torch.save(resnet34.state_dict(), "./logs/model/model_params.ckpt")
    load_model = resnet34.load_state_dict(torch.load("./logs/model/model_params.ckpt"))


def main():
    torch.manual_seed(11)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained_model()
    save_load_model()


if __name__ == '__main__':
    main()
