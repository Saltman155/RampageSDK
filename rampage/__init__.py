__all__ = [
    "npu_unique",
]

import os
import warnings

from .get_chip_info import Dsmi_dc_Func

import rampage._C


from .ops.npu_unique import npu_unique


def _set_env():
    rampage_root = os.path.dirname(os.path.abspath(__file__))

    # 默认配置
    customize = "customize"
    opapi_name = "libcust_opapi.so"

    try:
        dsmi = Dsmi_dc_Func()
        soc_version = dsmi.chip_version_h()
        # 判断是否为A5
        if soc_version and ("95" in soc_version or soc_version == "Ascend950PR"):
            customize = "customize_arch35"
            opapi_name = "libcust_opapi_arch35.so"
    except Exception:
        warnings.warn("Failed to get chip version, falling back to default logic.")

    # 根据 customize 选择路径
    rampage_opp_path = os.path.join(rampage_root, "packages", "vendors", customize)

    ascend_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if ascend_custom_opp_path:
        new_path = rampage_opp_path + ":" + ascend_custom_opp_path
    else:
        new_path = rampage_opp_path
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = new_path

    rampage_op_api_so_path = os.path.join(rampage_opp_path, "op_api", "lib", opapi_name)
    rampage._C._init_op_api_so_path(rampage_op_api_so_path)

_set_env()
