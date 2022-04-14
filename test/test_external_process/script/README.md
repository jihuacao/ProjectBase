# Script README
## run_test.sh
* 功能
  * 运行test_external_process所有测试用例
* 主要逻辑
  * 所有external_process都遵循以下规则进行划分
* 示例命令
  * 只运行gflags的测试
    * MinGW-**在cygwin中测试通过**
      ```shell
        bash ${rpath}/run_test.sh --toolchain="MinGW" --output_root="$(pwd)" --exclude="\*" --include="test_googleflags" 2>&1 1>$(pwd)/test.log
      ```
    * VS-Visual Studio 16 2019-**使用git bash测试通过**
      ```
      ```
    * LLVM-**使用cygwin中进行测试-失败**
      ```shell
        bash ${rpath}/run_test.sh --toolchain="LLVM" --output_root="$(pwd)" --exclude="\*" --include="test_googleflags" 2>&1 1>$(pwd)/test.log
      ```
