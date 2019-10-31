from darknet import *


def main():
    # global dev
    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Cuda: {}".format(torch.cuda.is_available()))

    # blocks = parse_cfg("cfg/yolov3.cfg")
    # print(create_modules(blocks))

    print("Loading network.....")
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("data/yolov3.weights")
    print("Network successfully loaded")

    # If there's a GPU availible, put the model on GPU
    if torch.cuda.is_available():
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    test_img = get_test_input("./data/imgs/dog-cycle-car.png")
    raw_pred = model(test_img, torch.cuda.is_available())
    print(raw_pred)
    pred = write_results(raw_pred, 0.5, 80, nms_conf=0.4)
    print(pred)


if __name__ == '__main__':
    main()
