import torch
import torch.nn as nn
from numpy import prod

net_config = {
    "conv2d": {
        "class": nn.Conv2d,
        "kwargs": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "inferFirst": True,
        },
    "batchNorm2d": {
        "class": nn.BatchNorm2d, 
        "kwargs": ["num_features"],
        "inferFirst": True,
        },
    "relu": {
        "class": nn.ReLU, 
        "kwargs": [],
        },
    "avgPool2d": {
        "class": nn.AvgPool2d, 
        "kwargs": ["kernel_size", "stride", "padding"],
        },
    "linear": {
        "class": nn.Linear, 
        "kwargs": ["in_features", "out_features","bias"],
        "inferFirst": True,
        },
    "batchNorm1d": {
        "class": nn.BatchNorm1d, 
        "kwargs": ["num_features"],
        "inferFirst": True,
        },
    "flatten": {
        "class": lambda: nn.Flatten(start_dim=-3), 
        "kwargs": [],
        },
}

def layer_from_string(layer_str, input_shape):
    args = layer_str.split(",")
    layer_type = args[0]
    assert layer_type in net_config.keys(), f"Unexpected layer type: {layer_type}"

    kwargs = {}
    if net_config[layer_type].get("inferFirst", False):
        values = [input_shape[0]]
    else:
        values = []
    if len(args)>1:
        values = values + args[1:]
    for i in range(len(values)):
        kwargs[net_config[layer_type]["kwargs"][i]] = int(values[i])

    layer = net_config[layer_type]["class"](**kwargs)
    
    x = torch.rand(input_shape)
    if layer_type.startswith("batchNorm"):
        output_shape = input_shape
    elif layer_type=="flatten":
        output_shape = (prod(input_shape),)
    else:
        output_shape = layer(x).shape

    return layer, output_shape

def net_from_string(string, input_shape, target_shape=None):
    # Given a string, return the corresponding torch.nn.Sequential
    # Separate between CNN and MLP parts
    strs = string.split("&")
    assert len(strs)==2, f"Expected CNN and MLP strings, separated by '&', but got: {string}"
    cnn_str, mlp_str = strs

    # Create CNN
    cnn_str = cnn_str.split(";")
    if cnn_str == [""]: cnn_str=[]
    layers = []
    for i in range(len(cnn_str)):
        layer, output_shape = layer_from_string(cnn_str[i], input_shape=input_shape)
        layers.append(layer)
        input_shape = output_shape


    mlp_str = mlp_str.split(";")
    if mlp_str==[""]: mlp_str=[]

    if target_shape is not None:
        assert len(target_shape)==1, "Target shape should be the output of a linear layer"
        mlp_str += [f"linear,{target_shape[0]},0"]
    
    if len(mlp_str)==0:
        return nn.Sequential(*layers), output_shape
    # layer, output_shape = layer_from_string("flatten", input_shape=input_shape)
    # layers.append(layer)
    # input_shape = output_shape
    mlp_str = ["flatten"]+mlp_str

    for i in range(len(mlp_str)):
        layer, output_shape = layer_from_string(mlp_str[i], input_shape=input_shape)
        layers.append(layer)
        input_shape = output_shape

    # if target_shape is not None:
    #     assert len(target_shape)==1, "Target shape should be the output of a linear layer"
    #     layer, output_shape = layer_from_string(f"linear,{target_shape[0]},0", input_shape=input_shape)
    #     layers.append(layer)
    
    return nn.Sequential(*layers), output_shape

def check_valid_string(string):

    strs = string.split(" & ")
    assert len(strs)==2, f"Expected CNN and MLP strings, separated by '&', but got: {string}"
    cnn_str, mlp_str = strs
    cnn_str = cnn_str.split(";")



if __name__=="__main__":
    # default_str = "conv2d,16,3,1,0 conv2d,9,3,1,0 batchNorm2d relu & linear,50 relu linear,25 relu"
    # default_str = " & linear,50 relu linear,25 relu"
    # default_str = "conv2d,16,3,1,0 relu conv2d,32,5,1,0 relu conv2d,1,1 & "
    default_str = "conv2d,10,3,1,0;relu;conv2d,10,5,1,0;relu;avgPool2d,5,1,0&"
    # default_str = "conv2d,16,3,1,0 conv2d,9,3,1,0 batchNorm2d relu & "

    input_shape = (3,12,12)

    net, out = net_from_string(default_str, input_shape=input_shape)#, target_shape=(1,))
    # res = net_from_string(default_str, input_shape=(3,9,9))

    v = torch.rand(10,*input_shape)
    y = net(v)
    
    print(net)