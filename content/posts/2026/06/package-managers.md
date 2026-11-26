---
title: "Windows 软件管理方案分享"
date: 2026-06-03T12:00:00+08:00
draft: false
featuredImg: ""
description: '聊聊本机的 Windows 软件管理思路，八个包管理器各司其职，再加上游戏平台和兜底目录'
tags:
  - 工具
  - Windows
  - 开发环境
author: BLESS
scrolltotop: true
toc: true
mathjax: false
comments: false
---

# Windows 软件管理方案分享

最近整理了一下 `updateall.ps1`，发现本机跑着八个包管理器，趁机把整体的软件管理思路梳理一下，不只是开发工具，也包括游戏和其他日常软件，免得以后搞混了装到错地方。

---

## 系统级

### Scoop

用了很久，体验最好的 Windows 包管理器，主要覆盖跨平台的命令行工具和开发工具链。最满意的地方是软件源的质量，很多在 Linux 上用惯的工具在 Scoop 里都能直接装到，比如 helix、yazi、git、btop、fzf、ripgrep、eza、zoxide、fd、jq……现在本机装了七十多个包，日常用到的命令行工具基本都在这里。

所有东西装在 `~/scoop/apps/`，完全不需要管理员权限，用 shims 挂到 PATH，卸载也干净，不会留下注册表残留。`nerd-fonts` bucket 管字体也很方便，FiraCode NF 和 Sarasa Gothic 都是从这里装的。

另外 winget 在 Windows 10 的低版本上可能没有预装，这时候可以直接用 Scoop 来装或者更新 winget 本身，算是 Scoop 作为基础设施的一个体现。

因为很多工具本身也是通过 Scoop 装的，所以更新脚本里 Scoop 必须单独跑在最前面，等它完成之后再并行更新其他东西。

### winget

微软官方出的，侧重于专供 Windows 的桌面应用和商业软件，这类东西在 Scoop 里基本找不到。包库来自 winget-pkgs 社区仓库，走官方渠道相对可信。

`winget upgrade --all` 是我觉得最实用的功能，系统里装的软件一条命令全部更新，比手动挨个去官网下安装包省心太多。还有 `winget export` 可以导出当前安装的所有软件列表，换新机的时候 `winget import` 一次性恢复，算是一个不错的迁移方案。

分工上比较清晰：命令行开发工具走 Scoop，Windows 桌面应用走 winget，两者重叠的情况不多。

### Chocolatey

本机的 Chocolatey 是通过 winget 装的，包库是三个里最大的，但维护质量参差不齐，而且基本都需要管理员权限。目前实际只用它装了一个东西：MSVC，Scoop 和 winget 都没收录，choco 的 `visualstudio2022buildtools` 包正好能覆盖这个需求。说白了就是为了这一个用途存在的。

### MSYS2

性质上和上面三个不太一样，MSYS2 提供的是一套完整的类 Unix 环境，内置 pacman 包管理器。在 Windows 上编译 C/C++ 项目或者需要 gcc、make、cmake 这类 GNU 工具的时候，MSYS2 是绕不开的选择。Git for Windows 底层用的也是这套环境。

更新时有个小细节，pacman 需要跑两遍：第一遍 `-Syyu` 强制刷新数据库同时升级 pacman 自身，第二遍 `-Syu` 才升级其余包。跑一遍有时候会漏掉东西，这个是 pacman 的设计，不是 bug。

MSYS2 提供了 MSYS2、MinGW-w64、UCRT64 几套不同的子系统，底层 ABI 不一样，和 Windows 原生工具链的互操作性也有区别，具体选哪个看项目需求，一般来说现代项目推荐 UCRT64。

### 游戏平台

游戏统一交给官方平台管理：Steam、Epic、GOG、Battle.net，各家自己维护更新，不走任何包管理器。这类软件本来就是平台强绑定的，没必要强行纳入 Scoop 或 winget 管理，官方渠道反而最省心。

### 兜底：固定安装目录

不在以上任何包管理器覆盖范围内的软件，统一安装到一个固定目录。这样不管过了多久，想找某个软件装在哪直接去那个目录翻，不用靠记忆，也不会在 C 盘到处散落。

---

## Python

### uv

Rust 写的，速度比 pip 快 10-100 倍，本机所有 Python 项目现在全走 uv 管理。之前同时在用 pip、venv、pyenv、pipx，现在一个 uv 全替代了，少维护好几套工具。

`uv run` 是我觉得体验提升最大的地方，不需要先 `source .venv/bin/activate` 激活虚拟环境，直接 `uv run python script.py` 就能在正确的环境里运行，少了很多上下文切换的心智负担。

`uv tool` 这个子命令值得单独说一下。有一类 Python 工具，官方文档推荐的安装方式是 `pip install`，但它们又不属于某个具体项目的依赖，需要全局可用——Scoop 里能找到的（比如 ruff）还是走 Scoop，但 Scoop 没有收录的就用 `uv tool install` 来管。

每个工具装在各自独立的虚拟环境里，入口暴露到 PATH，依赖完全隔离，不会互相污染，也不会搅乱项目环境。这和直接 `pip install --user` 装到全局 site-packages 的区别很大，后者各工具依赖混在一起，迟早出冲突。

```bash
uv tool install <package>     # 安装
uv tool upgrade --all         # 一次性升级所有工具
uv tool list                  # 查看已安装的工具
```

---

## 多语言运行时

### mise

mise 是运行时版本管理器，asdf 的 Rust 重写，原来叫 rtx。理论上能管很多语言的运行时，但本机实际上只用它做一件事：管理 Node.js 版本和全局 npm 包。Python 那边完全交给 uv，不需要 mise 插手。

最省心的地方是 shell hook，进入项目目录会自动读 `.mise.toml` 切换到对应 Node.js 版本，不需要手动激活。全局 npm 包（比如一些前端 CLI 工具）也通过 mise 统一管理，`mise upgrade` 一起更新。

---

## LaTeX

### MiKTeX

MiKTeX 是 Windows 上最常用的 LaTeX 发行版，包管理走内置的 `mpm`（MiKTeX Package Manager）。它的按需安装设计我挺喜欢：第一次编译遇到缺失宏包会自动下载，不需要提前把几个 GB 的包全装上，初装体积小很多。

批量更新用 `miktex packages update`，也有 MiKTeX Console 这个 GUI 工具，但我基本只在命令行里操作。

---

## Rust 工具链

### rustup

Rust 官方的工具链管理器，安装和更新 Rust 本身就靠它，没有替代品。`rustup update` 把所有已安装的 toolchain 更新到最新，非常简单。

rustup 还管 stable/beta/nightly 多版本并存，通过项目根目录的 `rust-toolchain.toml` 可以声明这个项目用哪个 channel，进去就自动切换，和 mise 的设计思路类似。跨平台编译也靠 `rustup target add` 来管理 target。

---

## 小结

八个工具，一张表理清楚：

| 包管理器       | 覆盖范围                   | 更新命令                      |
| -------------- | -------------------------- | ----------------------------- |
| Scoop          | Windows CLI 工具           | `scoop update *`              |
| winget         | Windows 桌面软件           | `winget upgrade --all`        |
| Chocolatey     | 老旧/企业 Windows 软件     | `sudo choco upgrade all -y`   |
| MSYS2 / pacman | Unix 工具链、MinGW 编译器  | `pacman -Syyu && pacman -Syu` |
| uv             | Python 项目/工具/解释器    | `uv tool upgrade --all`       |
| mise           | Node.js 版本 + 全局 npm 包 | `mise upgrade`                |
| MiKTeX / mpm   | LaTeX 宏包                 | `miktex packages update`      |
| rustup         | Rust 工具链                | `rustup update`               |

`updateall.ps1` 里 Scoop 单独串行跑完，其余七个并行，整体更新一次大概几分钟，比以前挨个手动执行省了不少时间。
