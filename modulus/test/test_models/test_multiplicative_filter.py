from modulus.models.multiplicative_filter_net import MultiplicativeFilterNetArch
import torch
import numpy as np
from modulus.key import Key


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    tri_names = ("frequency", "phase")
    for tri_name in tri_names:
        _dict["first_filter." + tri_name] = "fourier_filter_first_" + tri_name + ":0"
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["fc_layers." + str(i) + ".linear." + pt_name] = (
                "fc_" + str(i) + "/" + tf_name + ":0"
            )
        for tri_name in tri_names:
            _dict["filters." + str(i) + "." + tri_name] = (
                "fourier_filter_layer" + str(i) + "_" + tri_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


def test_multiplicative_filter():
    filename = "./test_models/data/test_multiplicative_filter.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = MultiplicativeFilterNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            if "filter" in _name:
                _tensor.data = torch.from_numpy(Wbs[name_dict[_name]])
            else:
                _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, atol=1e-4), "Test failed!"
    print("Success!")


if __name__ == "__main__":
    test_multiplicative_filter()
