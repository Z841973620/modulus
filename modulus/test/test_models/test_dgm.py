from modulus.models.dgm import DGMArch
import torch
import numpy as np
from modulus.key import Key


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    dgm_name = ["z", "g", "r", "h"]
    # start layer
    for pt_name, tf_name in names:
        _dict["fc_start.linear." + pt_name] = "fc_start/" + tf_name + ":0"
    # end layer
    for pt_name, tf_name in names[:2]:
        _dict["fc_end.linear." + pt_name] = "fc_end/" + tf_name + ":0"
    # middle layers
    for i in range(nr_layers - 1):
        for dn in dgm_name:
            _dict["dgm_layers." + str(i) + "." + dn + ".bias"] = (
                "dgm_" + dn + str(i + 1) + "/biases:0"
            )
            _dict["dgm_layers." + str(i) + "." + dn + ".linear_1.weight"] = (
                "dgm_" + dn + str(i + 1) + "/weights1:0"
            )
            _dict["dgm_layers." + str(i) + "." + dn + ".linear_2.weight"] = (
                "dgm_" + dn + str(i + 1) + "/weights2:0"
            )
            _dict["dgm_layers." + str(i) + "." + dn + ".linear_1.weight_g"] = (
                "dgm_" + dn + str(i + 1) + "/alphas1:0"
            )
            _dict["dgm_layers." + str(i) + "." + dn + ".linear_2.weight_g"] = (
                "dgm_" + dn + str(i + 1) + "/alphas2:0"
            )
    return _dict


def test_dgm():
    filename = "./test_models/data/test_dgm.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = DGMArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"
    print("Success!")


if __name__ == "__main__":
    test_dgm()
