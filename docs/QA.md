### Q1: 如何查看构建产物？

编译完成后，算子包位于 `rampage/packages/vendors/customize/` 目录下：
```
packages/vendors/customize/
├── op_proto/           # 算子原型库
├── op_impl/            # Tiling 库 + Kernel 实现 + 二进制
├── op_api/             # ACLNN API 库和头文件
├── framework/          # npu_supported_ops.json
└── version.info
```

### Q2: 如何支持新的芯片型号？

在 `kernels/op_host/xxx.cpp` 的 OpDef 中添加：
```cpp
this->AICore().AddConfig("ascend_new_chip");
```
同时在 `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT` 列表中添加新芯片标识。


### Q3: 如何为 Arch35 架构添加算子？

1. 在 `kernels_arch35/op_host/` 和 `kernels_arch35/op_kernel/` 下添加算子文件
2. 需要在 `kernels_arch35/CMakeLists.txt` 中添加对应的构建逻辑（参考 `kernels/CMakeLists.txt`）
3. 构建时设置 `USE_ARCH35=true`