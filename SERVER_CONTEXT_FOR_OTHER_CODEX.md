# 服务器协作环境说明（供其他 Codex / 其他项目使用）

## 基本协作方式

- Codex 不能直接登录或操作真实实验服务器。
- 由用户在本机、GitHub、服务器之间搬运代码、命令、少量日志、少量图片或截图。
- 给服务器执行的内容应尽量是：
  - GitHub 可 clone 的代码；
  - 明确的 shell 命令；
  - 可复制的最小日志回传指令；
  - 小体积结果文件或文本摘要。

## 服务器能力

- GPU：V100 32GB。
- 环境依赖安装自由，可以使用 `pip/conda/apt` 等安装项目所需依赖。
- 不需要为了兼容服务器而强行限制为 Python 标准库。

## 网络与 Git

- 服务器弱外网。
- `git clone` 通常可以成功。
- `git pull` 经常失败或不稳定。
- 推荐同步代码方式：
  - 首选：新建目录重新 `git clone`。
  - 或者删除旧目录后重新 `git clone`。
  - 避免把关键流程依赖在 `git pull` 上。
- 代码仓库应保持轻量，不要提交数据集、大模型权重、大量可视化图片或中间输出。

## 与 Codex 协作的建议格式

让 Codex 交付代码时，最好同时要求：

- 给出 GitHub 最新 commit hash。
- 给出服务器从零开始的运行命令。
- 给出失败时需要回传的最小诊断命令。
- 给出成功时需要回传的最小结果文件，例如：

```bash
sed -n '1,220p' path/to/result.md
tail -n 80 path/to/run.log
ls -lh path/to/output
```

## 推荐服务器执行模式

由于 `git pull` 不稳定，推荐类似：

```bash
cd /mnt/workspace/workgroup/leijian/benchmark
rm -rf project-new
git clone git@github.com:OWNER/REPO.git project-new
cd project-new
```

如果需要保留旧输出，不要删除数据目录或输出目录；只删除/替换代码目录。

## 注意事项

- 不要把包含服务器私有路径、数据说明、访问方式或人工搬运记录的本地说明文件提交到公共仓库。
- 如果必须让其他 Codex 知道服务器路径，建议单独提供一个不提交 Git 的本地 Markdown。
- 对需要长期复现的实验，优先让脚本输出 Markdown 摘要和 JSON/JSONL 机器可读结果。

