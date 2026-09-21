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

---

## 5. 每日 100 Credits 自动领取（签到）

服务启动后自动运行（随反代进程，无需单独脚本）。只要"当天还没领取"，无论几点启动、是否错过 10:00 刷新，都会自动补领；领取成功后当天不再重复，等到下一个刷新窗口（活动每日 10:00 UTC+8 刷新，窗口取 10:15，留 15 分钟余量）。活动一直未开放的话，最多查到 12:00 就当天放弃、次日窗口再看——不会整天轮询。

配置（任选一种，环境变量优先）：

1. 项目目录下 `checkin.json`（推荐；已加入 .gitignore，不会进版本库）：

```json
{
  "pat": "pt-你的Qoder_PAT",
  "retry_minutes": 30
}
```

多账号：`"pats": ["pt-一", "pt-二"]`

2. 环境变量：`QODER_CHECKIN_PAT`（多个用英文逗号分隔）、`QODER_CHECKIN_RETRY_MINUTES`（未领取时的重试间隔，默认 30 分钟）

说明：

- 认证复用桥自身会话（PAT → jobToken），无需额外配置
- 领取过程与结果打印在服务日志里（`[checkin]` 前缀），成功示例如：`[checkin] nickXXXX: 签到成功 +100 积分`
- 不配置则功能自动关闭，不影响主链路

---

## 6. 指标透传（缓存命中率 / 每秒 token 数）

上游在每次回答末尾会下发 usage（prompt / completion / total / `prompt_tokens_details.cached_tokens` / `credits`）。本桥原先把 usage 写死成 0，导致 Hermes 侧拿不到缓存命中率和每秒输出 token 数；现已改为真实透传：

- **流式**：在 `[DONE]` 之前补一个 OpenAI `stream_options.include_usage` 风格的 usage chunk（`choices: []`）
- **非流式**：响应体的 `usage` 直接来自上游
- `prompt_tokens_details.cached_tokens` 是客户端算缓存命中率的标准字段；`completion_tokens` 是算每秒 token 数的分子
- 上游没上报的字段不写入响应（避免把"没上报"伪装成 0）
- 额外带上上游的 `credits`（本次消耗积分），便于核对用量

实现：`usage.py`（提取 + 映射，纯函数、可单测）+ `transform.extract_usage_line`（信封解包）。

---

## 7. 上游"忙 / 排队"的处理

上游繁忙时会用 `code 10605` 之类的信封（常带 403）告知"忙 / 排队"。这类回应**不是鉴权失败**：

- 不再误判为 401 → 不再白刷新会话（旧行为会轮换 token，连累其他在途请求）
- 非流式：返回 **HTTP 503 + `Retry-After`**（上游给了 `retryAfterSeconds` 时），`error.type = "upstream_busy"`
- 流式：流内错误 chunk 的 `error.type = "upstream_busy"`，客户端可据此退避重试
- 真正的登录过期（`code 105` / 401 / 403 无忙标记）仍照旧刷新会话重试一次

实现：`qoder_auth.QoderBusyError` + `detect_upstream_busy()`（分类），桥接层负责状态码映射。
