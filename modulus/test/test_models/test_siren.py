from modulus.models.siren import SirenArch
import torch
import numpy as np
from modulus.key import Key


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases")]
    for i in range(nr_layers + 1):
        for pt_name, tf_name in names:
            _dict["layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    return _dict


def test_siren():
    filename = "./test_models/data/test_siren.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = SirenArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
        first_omega=params["first_omega"],
        omega=params["omega"],
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
    test_siren()
