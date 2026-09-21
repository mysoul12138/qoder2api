# Qoder2API (Hermes 专属部署配置指南)

本项目已成功部署至 `H:\qoder2api`，并完成了独立的 Python 虚拟环境与依赖构建。

---

## 1. 快速使用

### 启动服务
- **前台运行（查看实时日志）**：双击运行 `start.cmd`
- **后台静默运行（无黑框，推荐）**：双击运行 `start-silent.vbs`（日志自动保存在 `qoder2api.log`）

### 检查与停止
- **状态检查**：双击运行 `status.cmd`，自动检测 8963 端口并拉取模型列表
- **安全停止**：双击运行 `stop.cmd`，精准停止占用 8963 端口的进程（不会误杀系统其他 Python 进程）

---

## 2. 获取 Qoder 访问令牌 (PAT)

1. 打开浏览器登录 [Qoder 账号集成中心](https://qoder.cn/account/integrations)
2. 在「**服务集成**」中点击「**创建个人访问令牌**」（Personal Access Token）
3. 复制生成的令牌（形如 `pt-xxxx_019exxxx-...`）
4. 该令牌作为 API Key 即可鉴权使用。

---

## 3. 接入 Hermes 配置范例

若需要将 Qoder 提供的模型接入 Hermes，可在 Hermes 的 `config.yaml` 中增加 `custom_providers`：

```yaml
custom_providers:
  qoder:
    base_url: "http://127.0.0.1:8963/v1"
    api_key: "pt-xxxx_你的Qoder_PAT令牌"
    models:
      - Qwen3.8-Max-Preview
      - Qwen3.7-Max
      - Qwen3.7-Plus
      - DeepSeek-V4-Pro
```

---

## 4. 技术特性与 Windows 优化

- **协议兼容**：全面兼容 OpenAI `/v1/chat/completions` 与 `/v1/models`，支持流式 SSE、工具调用与推理。
- **环境隔离**：虚拟环境独立建立在 `H:\qoder2api\.venv`，完全不占用系统 C 盘。
- **Windows 代理避坑**：启动脚本内已内置 `set NO_PROXY=*`，彻底防止 Windows 下 httpx 读取 IE 系统代理导致的请求超时。
