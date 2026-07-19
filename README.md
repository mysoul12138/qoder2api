# qoder2api

把 [QoderWork](https://qoder.com.cn) 的私有聊天协议反向实现，封装成 **OpenAI 兼容 API**。
任何支持 OpenAI 接口的客户端，带上你的 Qoder PAT，即可调用 QoderWork 的全部模型额度。

- 纯 Python 实现（flask + httpx + pycryptodome），无 Node / wasm 依赖
- **模型列表动态加载**：从 Qoder 网关实时拉取，网关加减模型本服务自动同步，无需改代码
- 支持 SSE 流式 / 非流式、多模态（图片）、Tool Calls、深度推理（thinking）
- 多 PAT 并发隔离（每 PAT 独立会话 + 目录缓存）

---

## 工作原理

```
客户端 POST /v1/chat/completions (OpenAI 格式, Bearer PAT)
   │
   ▼
OpenAiBridge
   ├─ PAT → jobToken 冷交换 / refreshToken 周期续期 (qoder_auth)
   ├─ get_catalog() → GET gateway /api/v2/model/list  (动态, 600s 缓存)
   │     └─ resolve_model("GLM-5.2") → ("GLM-5.2","gm51model")
   ├─ transform: OpenAI messages → Qoder chat_context (套 baseprompt 模板)
   ├─ qoder_auth: 带 cosy 签名头 POST gateway SSE 流
   └─ transform: Qoder SSE 增量 → OpenAI data: chunks / 同步响应
   │
   ▼
客户端收到标准 OpenAI 响应
```

`qoder_auth.py` 用纯 Python 复刻了 `qoder_auth_wasm` 的全部加密/签名逻辑（自定义 base64、cosy 请求签名、RSA 密钥交换、AES payload 加密），**完全不依赖 wasm**。

---

## 支持的模型

模型列表由网关动态下发，以下为当前 `chat` 场景的实际可用模型（仅供参考，以 `/v1/models` 实时返回为准）：

| 模型名 (display_name) | 内部 key | 视觉 | 备注 |
|---|---|---|---|
| Qwen3.8-Max-Preview | `qmodel_preview` | ✅ | 最新基座，2.4T 参数 |
| Qwen3.7-Max | `qmodel_latest` | ✅ | **默认模型** |
| Qwen3.7-Plus | `qmodel` | ✅ | 进阶 |
| Qwen3.6-Flash | `q36fmodel` | ✅ | 轻量 |
| DeepSeek-V4-Pro | `dmodel` | ✅ | 专家 |
| DeepSeek-V4-Flash | `dfmodel` | ✅ | 轻量 |
| GLM-5.2 | `gm51model` | ✅ | 智谱 |
| Kimi-K2.7-Code | `kmodel` | ✅ | 256K |
| MiniMax-M2.7 | `mmodel` | ❌ | 非 VL |

视觉能力（`is_vl`）同样来自网关动态目录，非写死。

> `auto`（路由入口）不在列表中，因为它不是真实可选模型。

---

## 快速开始

### 获取 PAT（作为 API Key）

1. 打开 <https://qoder.cn/account/integrations>
2. 在「**服务集成**」里点**创建个人访问令牌**（Personal Access Token）
3. 复制生成的令牌，形如 `pt-xxxx_019exxxx-...`

这个令牌就当作 OpenAI 的 `api_key` 使用，后续所有请求通过 `Authorization: Bearer <PAT>` 携带。

### Docker 部署（推荐）

```bash
docker compose up -d --build
# 服务监听 http://localhost:8963
```

`docker-compose.yaml`：

```yaml
services:
  qoder2api:
    build: .
    platform: linux/amd64
    restart: unless-stopped
    ports:
      - "8963:8963"
    environment:
      QODER_HOST: 0.0.0.0
      QODER_PORT: 8963
```

### 本地运行

```bash
pip install -r requirements.txt   # flask, httpx, pycryptodome
python openai_bridge.py           # 默认 127.0.0.1:8963
```

> **Windows 本地注意**：httpx 会读取系统 IE 代理导致超时，本地跑需带 `NO_PROXY='*'`。
> 服务器（Linux 容器）无此问题。

---

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `QODER_HOST` | `127.0.0.1`（本地）/ `0.0.0.0`（容器） | 监听地址 |
| `QODER_PORT` | `8963` | 监听端口 |
| `QODER_SIGNATURE_SECRET` | 内置默认 | cosy 请求签名密钥（一般无需改） |

PAT **不**配置在环境里，由客户端每次请求通过 `Authorization: Bearer <PAT>` 提供，服务端按 PAT 哈希建立独立 Bridge 实例池。

---

## 使用示例

### 获取模型列表

```bash
curl http://localhost:8963/v1/models \
  -H "Authorization: Bearer <PAT>"
```

```json
{"object":"list","data":[
  {"id":"Qwen3.8-Max-Preview","object":"model","created":0,"owned_by":"qoder"},
  {"id":"Qwen3.7-Max","object":"model","created":0,"owned_by":"qoder"},
  ...
]}
```

> 带 PAT → 返回该账户动态拉取的目录；不带 PAT → 返回内置兜底表。

### 聊天补全（非流式）

```bash
curl http://localhost:8963/v1/chat/completions \
  -H "Authorization: Bearer <PAT>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2",
    "messages": [{"role":"user","content":"用一句话介绍你自己"}]
  }'
```

### 聊天补全（流式）

```bash
curl -N http://localhost:8963/v1/chat/completions \
  -H "Authorization: Bearer <PAT>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.7-Max",
    "messages": [{"role":"user","content":"数到5"}],
    "stream": true
  }'
```

返回标准 OpenAI SSE：`data: {chunk}` ... `data: [DONE]`。

### Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8963/v1",
    api_key="<PAT>",            # PAT 当作 api_key
)

# 非流式
resp = client.chat.completions.create(
    model="GLM-5.2",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)

# 流式
for chunk in client.chat.completions.create(
    model="Qwen3.7-Max",
    messages=[{"role": "user", "content": "讲个笑话"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

### Tool Calls / 多模态

OpenAI 标准的 `tools` / `tool_choice` 与 `image_url` 多模态输入均支持。
多模态会校验模型视觉能力：对非 VL 模型（如 MiniMax-M2.7）传图会返回 400。

---

## API 参考

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/v1/chat/completions` | Bearer PAT | OpenAI 兼容聊天接口（流式/非流式/工具/视觉） |
| GET | `/v1/models` | 可选 PAT | 模型列表（带 PAT 动态，不带走兜底） |

---

## 项目结构

```
qoder2api/
├── openai_bridge.py        # Flask 入口 + OpenAiBridge(会话/流式转发) + 路由
├── qoder_auth.py           # Qoder 认证/签名/HTTP 客户端 (纯 Python 复刻 wasm)
├── transform.py            # OpenAI ↔ Qoder 消息转换 + 流式累积器
├── models.py               # 模型目录: 动态解析 + 内置兜底表
├── baseprompt.json         # Qoder 请求体模板
├── test_openai_bridge.py   # 单元测试
├── Dockerfile              # python:3.12-slim
├── docker-compose.yaml
└── requirements.txt        # flask, httpx, pycryptodome
```

---

## 动态模型加载机制

`OpenAiBridge.get_catalog()` 维护一个 **TTL 600 秒**的内存缓存：

1. 缓存命中 → 直接返回
2. 缓存过期 → `GET {gateway}/algo/api/v2/model/list?Encode=1`（带 cosy 鉴权）
3. 响应是明文 JSON（实测，与 chat SSE 端点一致的 `Encode=1` 行为）
4. `models.extract_catalog()` 解析 `chat` 场景，提取 `{display_name → key}` + `is_vl`
5. **拉取失败 → 显式 WARN 日志 + 回退内置兜底表**（不静默）

```python
# models.py 兜底表（仅动态拉取失败时用）
DEFAULT_MODEL_MAP = { "Qwen3.7-Max": "qmodel_latest", ... }  # 9 个
```

日志里出现 `[models] dynamic catalog loaded: N models` = 动态生效；
出现 `[models] WARN dynamic fetch failed (...); using fallback` = 走兜底（看错误原因）。

> **网关日后上线新模型**：本服务无需改代码，最多 10 分钟缓存窗口后自动出现在 `/v1/models`。

---

## 测试

```bash
python -m unittest test_openai_bridge -v   # 19 个测试, 无需联网/PAT
```

覆盖：模型解析、视觉能力判定、消息转换、流式累积、tool calls、错误 chunk、路由鉴权等。

---

## 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `401 Unauthorized` | PAT 失效或格式错；重新生成 |
| `Unsupported model 'xxx'` | 模型名不在动态目录；`curl /v1/models` 看实际可用列表 |
| `Image input is not supported` | 该模型非 VL；换带视觉的模型（见上表 ✅） |
| 本地 httpx 超时 | Windows 走系统代理；`NO_PROXY='*'` 直连 |
| 模型列表不更新 | 看日志是 `dynamic catalog loaded` 还是 `using fallback`；后者查网络/鉴权 |

---

## 技术栈

- **Python 3.12** · **Flask** · **httpx** · **pycryptodome**
- **Docker** 容器化部署

## 许可证

仅供学习研究。使用本服务产生的账号风险（额度、合规等）由使用者自负。
