from cnn_model.vggnet_model import get_model
width = 224
height = 224
channels = 3
model = get_model(pre_weight='', input_size=(width, height, channels))
