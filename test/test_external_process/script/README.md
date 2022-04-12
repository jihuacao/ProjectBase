# Script README
## run_test.sh
* 功能
  * 运行test_external_process所有测试用例
* 主要逻辑
  * 所有external_process都遵循以下规则进行划分
* 示例命令
  * 只运行gflags的测试
    * Cygwin
      ```shell
          bash test/test_external_process/script/run_test.sh --output_root="/cygdrive/d/workspace/test_external_process_all" --exclude="\*" --include="test_googleflags" 2>&1 1>/cygdrive/d/workspace/test_external_process_all/test.log
      ```
    * 使用git bash的MINGW系统进行windows的编译
      ```
      ```